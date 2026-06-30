struct Izhikevich <: NeuronModel
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    I_ext::Float64 # Baseline external current
end

# indexing function for packed voltage and recovery data
@inline get_voltage(u, idx) = u[(idx-1)*2 + 1]
@inline get_recovery(u, idx) = u[(idx-1)*2 + 2]
@inline voltage_index(i) = (i-1)*2 + 1
@inline function indexing(i)
    # Determine the variable slices for node i dynamically based on model type
    vars_per_node =  2
    idx_start = (i-1)*2 + 1
    return idx_start:(idx_start + vars_per_node - 1)
end

IzhikevichRS(I_ext=5.0) = Izhikevich(0.02, 0.2, -65.0, 8.0, I_ext) # standard Regular Spiking (RS) neuron

function local_dynamics!(du, u, model::Izhikevich, coupling_input)
    v = get_voltage(u, 1)
    u_rec = get_recovery(u, 1)  # recovery variable
    # Izhikevich standard equations for coupling current, see https://www.izhikevich.org/publications/spikes.htm
    du[1] = 0.04 * v^2 + 5.0 * v + 140.0 - u_rec + model.I_ext + coupling_input
    du[2] = model.a * (model.b * v - u_rec)
end

# coupling for Izhikevich with synaptic current based on neighbors' voltage
function coupling(u_i, neighbors, u_all, coupling_strength, ::Izhikevich)
    v_i = u_i[1]
    coupling_input = 0.0
    for j in neighbors
        v_j =  get_voltage(u_all, j) 
        # simple electrical/gap-junction coupling, or chemical synapse approximation
        coupling_input += (v_j - v_i)
    end
    #return (coupling_input / length(neighbors)) * coupling_strength
    return coupling_input * coupling_strength
end

function iz_discrete_condition(u, t, integrator)
    p = integrator.p
    # test if some neuron spiked; if any spiked (crossed 30.0 mV) then the reset callback is triggered
    for i in 1:nv(p.graph)
        if get_voltage(u, i) >= 30.0 
            return true
        end
    end
    return false
end

function iz_discrete_reset!(integrator)
    p = integrator.p
    models = p.models
    for i in 1:nv(p.graph)
        idx_v = voltage_index(i)
        idx_u = idx_v + 1
        # if this neuron spiked then reset
        if integrator.u[idx_v] >= 30.0
            model = models[i]
            # using the Izhikevich update equation
            integrator.u[idx_v] = model.c  # Reset voltage
            integrator.u[idx_u] += model.d # Step up recovery variable
        end
    end
end

function iz_network_dynamics!(du, u, p::DynamicalNetwork, t)
    g = p.graph
    models = p.models
    K = p.K
    for i in 1:nv(g)
        neighbors_i = neighbors(g, i)
        idx = indexing(i)
        coupling_input = coupling(view(u, idx), neighbors_i, u, K, models[i])
        local_dynamics!(view(du, idx), view(u, idx), models[i], coupling_input)
    end
end

function izhikevich_3node_network_test(K, models; T = 100)
    g = path_graph(3)
    add_edge!(g, 3, 1)

    u0 = [-65.0, -13.0, -60.0, -12.0, -65.0, -13.0]
    network = DynamicalNetwork(g, models, K)
    tspan = (0.0, T)

    iz_reset_callback = DiscreteCallback(iz_discrete_condition, iz_discrete_reset!)
    prob = ODEProblem(iz_network_dynamics!, u0, tspan, network)
    # Tsit5() is an adaptive solve; the 0.04*v^2 term in Izhikevich equations can cause du/dt to blow up
    sol = solve(prob, Tsit5(), callback=iz_reset_callback, reltol=1e-6, abstol=1e-6, dtmax=0.5)

    v1 = sol[1, :]
    v2 = sol[3, :]
    v3 = sol[5, :]

    plot(sol.t, [v1 v2 v3],
        xlabel="Time (ms)",
        ylabel="Membrane Potential v (mV)",
        title="Izhikevich network synchronization K=$K",
        label=["node 1" "node 2" "node 3"],
        lw=1.5)
end

# --- order parameter for spike data---
function order_parameter(sol, N, t_start)
    # throw out transients
    t_mask = sol.t .>= t_start
    ts = sol.t[t_mask]
    if length(ts) < 10 return 0.0 end
    
    # reconstruct phases for all nodes over time: a proxy in dense spiking regimes is normalized voltages
    data_R = Float64[]
    for tidx in 1:length(ts)
        u_t = sol.u[findfirst(x -> x == ts[tidx], sol.t)]

        # approximate phase based on geometric state space location
        phases = [atan(get_recovery(u_t, i), get_voltage(u_t, i ) ) for i in 1:N]
        
        # calculate Kuramoto Order Parameter at this time step
        R = abs(sum(exp(im * θ) for θ in phases)) / N
        push!(data_R, R)
    end
    return mean(data_R)
end





function iz_hysteresis_sweep_forward_backward(g, models, Ks, T, order_proxy; plt = nothing)
    N = nv(g)
    R_forward = Float64[]
    R_backward = Float64[]
    t_total::Float64 = T
    t_burn = t_total/3

    tspan = (0.0, t_total)
    iz_callback = DiscreteCallback(iz_discrete_condition, iz_discrete_reset!)

    println("fororward...")
    # initial resting state
    current_u0 = repeat([-65.0, -13.0], N)
    flattened_u0 = reduce(vcat, current_u0)

    # capture everything needed
    function evolve(u_initial, K)
        network = DynamicalNetwork(g, models, K)
        prob = ODEProblem(iz_network_dynamics!, u_initial, tspan, network)
        sol = solve(prob, Tsit5(), callback=iz_callback, reltol=1e-5, abstol=1e-5, dtmax=0.5)
        #R = order_parameter_proxy(sol, N, t_burn)
        #R = spike_phases(sol, N, t_burn)
        R = order_proxy(sol, N, t_burn)
        evolved_u = sol.u[end]

        return (evolved_u, R)
    end

    for K in Ks
        (flattened_u0, R) = evolve(flattened_u0, K)
        push!(R_forward, R)
    end

    println("backwards...")
    # backward sweep from forward evolved synchronized states
    for K in reverse(Ks)
        (flattened_u0, R) = evolve(flattened_u0, K)
        push!(R_backward, R)
    end
    reverse!(R_backward) # reverse to match Ks order
    return (R_forward, R_backward)
end


function iz_network_degree_correlated_external_current(g; I_base = 3.0, alpha = 1.2)
    N = nv(g)
    return [IzhikevichRS(I_base + alpha * degree(g, i)) for i in 1:N]
end


function iz_network_randomized_current(g; I_base = 3.0, alpha = 1.2)
    N = nv(g)
    return [IzhikevichRS(I_base + alpha * rand()) for i in 1:N]
end



