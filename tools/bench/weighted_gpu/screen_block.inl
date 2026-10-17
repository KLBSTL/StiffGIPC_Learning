const auto screen_start=Clock::now();
const int last=output["last_quality_iteration"].get<int>();
const bool cooling=current_cooldown&&policy_schedule.cooling_down();
std::optional<gipc::research::RecursiveNorm> norm;
const bool need_norm=!current_cooldown||policy_schedule.needs_recursive_norm(iteration,limit);
if(need_norm) {
    const double rr=dot(r.value,r.value);
    balanced_require(std::isfinite(rr)&&rr>=0,"Invalid recursive norm");
    norm=gipc::research::RecursiveNorm{rr,static_cast<double>(rhs_norm2)};
    point["recursive_norm_rr"]=rr;point["recursive_norm_evaluated"]=true;
    point["recursive_relative_l2"]=std::sqrt(rr/static_cast<double>(rhs_norm2));
    output["screen_checks"]=output["screen_checks"].get<int>()+1;
}
if(current_cooldown) {
    balanced_require(policy_schedule.last_check()==last,"Policy/native last-check disagreement");
    due=policy_schedule.due(iteration,limit,norm);
} else due=native_quality_check_due(iteration,last,norm->rr,norm->bb,limit);
output["schedule_queries"]=output["schedule_queries"].get<int>()+1;
point["screened_out"]=!due;
point["forced_quality_check"]=last==0||iteration==limit||iteration-last>=(cooling?2:8);
output["screen_host_ms"]=output["screen_host_ms"].get<double>()+milliseconds(screen_start);
