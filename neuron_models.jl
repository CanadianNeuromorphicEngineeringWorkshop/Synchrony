



using DifferentialEquations
using Graphs
using ComponentArrays
using LinearAlgebra
using Statistics
using Plots

abstract type NeuronModel end

include("kuramoto.jl")
include("izhikevich_model.jl")


