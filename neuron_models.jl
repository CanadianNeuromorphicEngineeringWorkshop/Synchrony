addPkg = false
if addPkg 
    using Pkg
    Pkg.add(["DifferentialEquations", "Graphs", "ComponentArrays", "LinearAlgebra", "Plots"])
end

using DifferentialEquations
using Graphs
using ComponentArrays
using LinearAlgebra
using Statistics
using Plots

abstract type NeuronModel end

struct Kuramoto <: NeuronModel
    omega::Float64  # frequency
end

function local_dynamics!(du, u, model::Kuramoto, coupling_input)
    du[1] = model.omega + coupling_input
end

struct DynamicalNetwork{G<:AbstractGraph, M<:AbstractVector{<:NeuronModel}}
    graph::G
    models::M
    K::Float64 # coupling
end

function coupling(node, neighbors, u, coupling_strength)::Float64
    coupling_input = 0.0
    for j in neighbors
        coupling_input += sin(u[j] - node[1])
    end
    coupling_input*coupling_strength 
end 

function network_dynamics!(du, u, p::DynamicalNetwork, t)
    g = p.graph
    models = p.models
    K = p.K
    for i in 1:nv(g)
        neighbors_i = neighbors(g, i)
        coupling_input = coupling(view(u, i:i), neighbors_i, u, K) 
        # compute the derivate to update du
        local_dynamics!(view(du, i:i), view(u, i:i), models[i], coupling_input)
    end
end

include("izhikevich_model.jl")
