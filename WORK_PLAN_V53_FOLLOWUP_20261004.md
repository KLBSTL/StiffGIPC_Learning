# v53 fixed-implementation follow-up

No solver edits or rebuilds. Preserve the v53 manifest, binary, runner, and earlier failures. All new simulations start at frame zero; GPU jobs run serially. Completion studies remain diagnostic, not performance certification.

1. Graph repeat: 100 frames / 600 s, same v53 settings as bunny100_completion, name bunny100_confirm.
2. Host control: 100 frames / 600 s, same settings except execution mode, name bunny100_host_confirm. Keep failures without increasing budgets.
3. Verify source/binary/runner identities, native exit and initial-guess selection, execution mode and PCG guards; independent accepted-path CCD for each exported trajectory.
4. Check frozen Stiff base material/scene compatibility and run one matching 100-frame / 600 s physical-reference observation if the available baseline supports it. Do not label baseline trajectory as physical ground truth or invent tolerance after seeing results. Explicitly distinguish endpoint CCD from accepted-substep coverage.
5. Independently measure FEM J, inversion counts, cloth edge stretch, normalized trajectory differences; render and inspect shared keyframes from actual data. Report limitations and choose the next action from evidence. Keep v53 default OFF until repeatability and physical quality are supported.

Update after Graph confirmation: 100/100 completed in 444.313 s, but whole-trajectory repeat RMS differs (max about 3.17% cloth scale, 4.33% FEM scale, 5.83% ABD scale). Add one identical base repeat to measure the base's own variation before attributing all differences to TOI or Graph. Same 100 frames / 600 s, unchanged base executable and inputs. This is a two-run observation, not a statistical guarantee or converged physical reference.

Final: all four new runs completed 100 frames; Graph 444.313 s, host 501.359 s, base 31.985 s, base repeat 33.109 s. Accepted-path CCD coverage 603/633/699/705 respectively, zero flags. Native v53 audits, source/binary provenance and identical scene/initial-state checks passed. Geometry and keyframe inspection show material trajectory differences; base repeat variation is much smaller. Final report V53_FOLLOWUP_20261004.md. This follow-up is complete, while physical quality/equivalence/performance acceptance remains unmet. Next priorities are a refined physical reference and earliest same-state divergence analysis, not another solver candidate. All simulation processes exited.
