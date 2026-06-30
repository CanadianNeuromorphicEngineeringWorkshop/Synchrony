addPkg = false
if addPkg 
    using Pkg
    Pkg.add(["DifferentialEquations", "Graphs", "ComponentArrays", "LinearAlgebra", "Plots"])
end

module Synchrony




include("neuron_models.jl")


export DynamicalNetwork, Kuramoto, network_dynamics!


end