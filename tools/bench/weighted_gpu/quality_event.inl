const int checked_iteration=point.at("iteration");
const bool override=output.value("zero_rho_true_quality_checked",false);
const bool repeated=output["last_quality_iteration"].get<int>()==checked_iteration;
const bool cooling_before=current_cooldown&&policy_schedule.cooling_down();
if(current_cooldown)policy_schedule.observe(checked_iteration,std::min<int>(max_iter-1,2000),l2,weighted,override&&repeated);
const bool cooling_after=current_cooldown&&policy_schedule.cooling_down();
point["policy_cooldown_after"]=cooling_after;
output["quality_events"].push_back({{"iteration",checked_iteration},{"true_relative_l2",l2},
    {"common_native_MAS_relative",weighted},{"zero_override",override},{"repeat",repeated},
    {"cooldown_before",cooling_before},{"cooldown_after",cooling_after},{"accepted",l2<=.0095&&weighted<=.0095}});
