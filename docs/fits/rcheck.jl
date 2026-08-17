# Quick R_t sense-check: fit a short window, extract Z_t via model conditioning.
using EpiSewer, Turing, Distributions, Statistics
import ComposableTuringIDModels as CT
using Turing.DynamicPPL: condition

d = EpiSewer.example_data()
sub = 1:50
y = d.measurements.concentration[sub]
flow = Vector{Float64}(d.flows.flow[sub])
# `n` is the length of the INFECTION series: the observed days plus the
# observation chain's lead-in, so every observation is scored.
n = length(y) + EpiSewer.observation_lead_in(EpiSewer.model())

mdl = CT.as_turing_model(EpiSewer.model(), (y = y, flow = flow), n)
chn = sample(mdl, NUTS(0.9; max_depth = 12), MCMCThreads(), 50, 2; warmup = 50, progress = false)

# Recover Z_t (log R_t) by conditioning the model on a posterior draw and
# evaluating forward (the same flat-VarName conditioning the plots worker
# validated). IDModel returns (; generated_y_t, expected_y_t, I_t, Z_t).
function condition_draw(draw, chain, chn, mdl)
    return Turing.DynamicPPL.condition(
        mdl,
        (
            rw_init = chn[:rw_init][draw, chain],
            std = chn[:std][draw, chain],
            Symbol("ϵ_t") => chn[Symbol("ϵ_t")][draw, chain],
            init_incidence = chn[:init_incidence][draw, chain],
            Symbol("Ascertainment.intercept") =>
                chn[Symbol("Ascertainment.intercept")][draw, chain],
            Symbol("σ") => chn[:σ][draw, chain],
        ),
    )
end

Z_draws = Vector{Vector{Float64}}()
for di in 1:min(size(chn, 1), 20)
    out = condition_draw(di, 1, chn, mdl)()
    push!(Z_draws, Vector{Float64}(out.Z_t))
end
Zmat = hcat(Z_draws...)
Rmed = exp.(vec(median(Zmat; dims = 2)))
println("RCHECK median over series: ", round(median(Rmed); digits = 3))
println("RCHECK range: ", round(minimum(Rmed); digits = 3), " - ", round(maximum(Rmed); digits = 3))
println("RCHECK head (1-8): ", round.(Rmed[1:min(8, end)]; digits = 2))
