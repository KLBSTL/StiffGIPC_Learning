#pragma once
#include <gipc/utils/json.h>
#include <core/accel_features.h>
#include <solver/mas_restrict_options.h>
#include <solver/legacy_restrict_options.h>
#include <solver/mas_factor_action_options.h>
#include <solver/ipc_options.h>
#include <solver/mas_fused_dot_options.h>
#include <solver/spmv_quadratic_options.h>
#include <collision/discrete_bvh.h>
#include <collision/ipc_contact_pool.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gipc
{
inline bool toi_history_flag(const char* name, bool fallback)
{
    const char* value=std::getenv(name);
    if(!value)return fallback;
    if(std::string(value)=="0")return false;
    if(std::string(value)=="1")return true;
    throw std::runtime_error(std::string(name)+" must be 0 or 1");
}

inline bool toi_flag_one(const char* name)
{
    const char* value=std::getenv(name);
    return value && std::string(value)=="1";
}

inline bool toi_flag_unless_zero(const char* name)
{
    const char* value=std::getenv(name);
    return !value || std::string(value)!="0";
}

struct ToiOptions
{
    std::string policy;
    bool robust_port=false;
    bool clamp_trial=false,clamp_safe=false;
    bool persist_contacts=false,keep_lambda=false,keep_gamma=false;
    bool independent_friction_history=false,keep_friction=false;
    bool legacy_volume=false,reuse_initial=true;
    double trial_velocity_tol=0;
    bool velocity_stop=false;
    std::string inner_exit;
    double remaining_fraction_tol=0;
    std::string mu_scope,mu_mode,mu_coordinates;
    double mu_scale=1;
    int stall_window=50;
    bool hold_delta=false,movement_exit=true;
    bool choose_start=false,restart_guard=false;
    int exit_probe_frame=0,exit_probe_outer=-1;
    int blocker_from=0,blocker_to=INT_MAX;

    static ToiOptions from_environment(double newton_tol,bool material_requires_noninversion)
    {
        ToiOptions result;
        result.policy=std::getenv("GIPC_TOI_POLICY")?std::getenv("GIPC_TOI_POLICY"):"paper";
        if(result.policy!="paper"&&result.policy!="robust")
            throw std::runtime_error("TOI policy must be paper or robust");
        result.robust_port=result.policy=="robust";
        result.clamp_trial=toi_flag_one("GIPC_TOI_TRIAL_INJECTIVITY");
        result.clamp_safe=material_requires_noninversion;
        if(const char* mode=std::getenv("GIPC_TOI_SAFE_INJECTIVITY"))
        {
            const std::string value(mode);
            if(value=="0")result.clamp_safe=false;
            else if(value=="1")result.clamp_safe=true;
            else if(value!="auto")throw std::runtime_error("Safe injectivity mode must be auto, 0, or 1");
        }
        const bool legacy_persist=toi_history_flag("GIPC_TOI_PERSIST_CONTACTS",false);
        // Keep the original short circuit: robust policy overrides even an
        // explicitly supplied reset. Expose that override in resolved JSON.
        result.persist_contacts=result.robust_port||toi_history_flag("GIPC_TOI_WARM_CONTACTS",legacy_persist);
        result.keep_lambda=result.robust_port||toi_history_flag("GIPC_TOI_WARM_LAMBDA",legacy_persist);
        result.keep_gamma=result.robust_port||toi_history_flag("GIPC_TOI_WARM_GAMMA",legacy_persist);
        result.independent_friction_history=toi_history_flag("GIPC_TOI_INDEPENDENT_FRICTION_HISTORY",false);
        result.keep_friction=toi_history_flag("GIPC_TOI_WARM_FRICTION",legacy_persist);
        if(result.robust_port&&result.independent_friction_history)
            throw std::runtime_error("Robust port uses a frame snapshot, not the friction history proxy");
        if((result.keep_lambda||result.keep_gamma||result.keep_friction)&&!result.persist_contacts)
            throw std::runtime_error("Warm lambda/gamma/friction requires retained contact identities");
        if(std::getenv("GIPC_TOI_WARM_FRICTION")&&!result.independent_friction_history)
            throw std::runtime_error("Independent friction warm option requires independent friction history");
        result.legacy_volume=std::getenv("GIPC_TOI_VOLUME_BOUND")
            &&std::string(std::getenv("GIPC_TOI_VOLUME_BOUND"))=="legacy";
        result.reuse_initial=toi_flag_unless_zero("GIPC_TOI_REUSE_INITIAL_ASSEMBLY");
        result.trial_velocity_tol=result.robust_port?.05:0;
        if(const char* value=std::getenv("GIPC_TOI_ROBUST_VELOCITY_TOL"))
        {
            result.trial_velocity_tol=std::stod(value);
            if(!std::isfinite(result.trial_velocity_tol)||result.trial_velocity_tol<=0)
                throw std::runtime_error("Robust trial velocity tolerance must be positive");
        }
        result.velocity_stop=result.robust_port||toi_flag_one("GIPC_TOI_ROBUST_VELOCITY_STOP");
        result.inner_exit=std::getenv("GIPC_TOI_INNER_EXIT")?std::getenv("GIPC_TOI_INNER_EXIT"):"native";
        if(result.inner_exit!="native"&&result.inner_exit!="velocity_only"&&result.inner_exit!="full_step_only")
            throw std::runtime_error("Unknown diagnostic inner exit policy");
        if(result.velocity_stop&&result.trial_velocity_tol==0)
            throw std::runtime_error("Robust velocity stop requires a positive trial velocity tolerance");
        result.remaining_fraction_tol=newton_tol;
        if(const char* value=std::getenv("GIPC_TOI_REMAINING_FRACTION_TOL"))
        {
            result.remaining_fraction_tol=std::stod(value);
            if(!std::isfinite(result.remaining_fraction_tol)||result.remaining_fraction_tol<=0||result.remaining_fraction_tol>=1)
                throw std::runtime_error("TOI remaining fraction tolerance must be between 0 and 1");
        }
        result.mu_scope=std::getenv("GIPC_TOI_MU_SCOPE")?std::getenv("GIPC_TOI_MU_SCOPE"):
            (result.robust_port?"movable":"full");
        if(result.mu_scope!="full"&&result.mu_scope!="movable")
            throw std::runtime_error("TOI mu scope must be full or movable");
        if(const char* value=std::getenv("GIPC_TOI_MU_SCALE"))result.mu_scale=std::stod(value);
        result.mu_mode=std::getenv("GIPC_TOI_MU_MODE")?std::getenv("GIPC_TOI_MU_MODE"):"diagonal";
        if(result.mu_mode!="diagonal"&&result.mu_mode!="mass")
            throw std::runtime_error("TOI stiffness mode must be diagonal or mass");
        result.mu_coordinates=std::getenv("GIPC_TOI_MU_COORDINATES")?std::getenv("GIPC_TOI_MU_COORDINATES"):
            (result.mu_mode=="diagonal"&&result.mu_scope=="movable"?"world_block":"generalized");
        if(result.mu_coordinates!="generalized"&&result.mu_coordinates!="world_block")
            throw std::runtime_error("TOI mu coordinates must be generalized or world_block");
        if(result.mu_coordinates=="world_block"&&(result.mu_mode!="diagonal"||result.mu_scope!="movable"))
            throw std::runtime_error("TOI world penalty requires diagonal mode and movable scope");
        if(const char* value=std::getenv("GIPC_TOI_STALL_WINDOW"))
        {
            result.stall_window=std::stoi(value);
            if(result.stall_window<=0)throw std::runtime_error("TOI stall window must be positive");
        }
        result.hold_delta=toi_flag_one("GIPC_TOI_STALL_HOLD_DELTA");
        result.movement_exit=toi_flag_unless_zero("GIPC_TOI_EARLY_TERMINATION");
        result.choose_start=toi_flag_one("GIPC_TOI_CHOOSE_START");
        result.restart_guard=toi_flag_one("GIPC_TOI_RESTART_FULL_STEP_GUARD");
        // One selected subproblem, diagnostic only. Keep the production
        // velocity tolerance and use a smaller fixed iteration budget.
        if(const char* value=std::getenv("GIPC_TOI_FULL_STEP_EXIT_PROBE"))
        {
            const std::string target(value);const auto colon=target.find(':');
            if(colon==std::string::npos||target.find(':',colon+1)!=std::string::npos)
                throw std::runtime_error("Exit probe requires frame:outer");
            size_t nf=0,no=0;
            result.exit_probe_frame=std::stoi(target.substr(0,colon),&nf);
            result.exit_probe_outer=std::stoi(target.substr(colon+1),&no);
            if(!result.robust_port||nf!=colon||no!=target.size()-colon-1
                ||result.exit_probe_frame<1||result.exit_probe_outer<0)
                throw std::runtime_error("Invalid robust exit probe target");
        }
        if(const char* value=std::getenv("GIPC_TOI_BLOCKER_AUDIT_FROM"))result.blocker_from=std::stoi(value);
        if(const char* value=std::getenv("GIPC_TOI_BLOCKER_AUDIT_TO"))result.blocker_to=std::stoi(value);
        if(result.blocker_from<0||result.blocker_to<result.blocker_from)
            throw std::runtime_error("Invalid TOI blocker audit frame window");
        return result;
    }

    Json to_json() const
    {
        Json result={{"policy",policy},{"robust_port",robust_port},
            {"trial_injectivity_clamp",clamp_trial},{"safe_injectivity_clamp",clamp_safe},
            {"contacts_retained",persist_contacts},{"lambda_retained",keep_lambda},{"gamma_retained",keep_gamma},
            {"friction_mode",robust_port?"frame_snapshot_previous_lambda":independent_friction_history?"independent_proxy":"coupled_al"},
            {"friction_history_retained",independent_friction_history?Json(keep_friction):Json(nullptr)},
            {"safe_volume_bound_mode",legacy_volume?"legacy":"normalized"},{"reuse_initial_assembly",reuse_initial},
            {"trial_velocity_tol_m_s",trial_velocity_tol},{"velocity_stop",velocity_stop},{"inner_exit",inner_exit},
            {"remaining_fraction_tol",remaining_fraction_tol},
            {"remaining_fraction_tol_source",std::getenv("GIPC_TOI_REMAINING_FRACTION_TOL")?"GIPC_TOI_REMAINING_FRACTION_TOL":"inherited_ipc_newton_tol"},
            {"mu_scope",mu_scope},{"mu_mode",mu_mode},{"mu_scale",mu_scale},{"mu_coordinates",mu_coordinates},
            {"mu_estimator",mu_coordinates=="world_block"?"0.1*max(movable FEM diagonal, diag((J H_ABD^-1 J^T)^-1))":"historical"},
            {"stall_window",stall_window},{"stall_hold_delta",hold_delta},{"movement_exit",movement_exit},
            {"choose_start",choose_start},{"restart_full_step_guard",restart_guard},
            {"full_step_exit_probe",exit_probe_frame?Json{{"frame",exit_probe_frame},
                {"outer",exit_probe_outer},{"max_inner",8}}:Json(nullptr)},
            {"choose_start_effective_for_policy",choose_start&&robust_port},
            {"restart_guard_requires_safe_restart",true},
            {"reduced_slack",toi_flag_one("GIPC_TOI_REDUCED_SLACK")},
            {"blocker_from",blocker_from},{"blocker_to",blocker_to}};
        result["requested_environment"]=Json::object();
        for(const char* name:{"GIPC_TOI_POLICY","GIPC_TOI_PERSIST_CONTACTS","GIPC_TOI_WARM_CONTACTS",
            "GIPC_TOI_WARM_LAMBDA","GIPC_TOI_WARM_GAMMA","GIPC_TOI_WARM_FRICTION",
            "GIPC_TOI_INDEPENDENT_FRICTION_HISTORY","GIPC_TOI_ROBUST_VELOCITY_TOL","GIPC_TOI_ROBUST_VELOCITY_STOP",
            "GIPC_TOI_REMAINING_FRACTION_TOL","GIPC_TOI_INNER_EXIT","GIPC_TOI_MU_SCOPE","GIPC_TOI_MU_MODE",
            "GIPC_TOI_MU_SCALE","GIPC_TOI_MU_COORDINATES","GIPC_TOI_CHOOSE_START","GIPC_TOI_RESTART_FULL_STEP_GUARD","GIPC_TOI_FULL_STEP_EXIT_PROBE",
            "GIPC_TOI_DELTA","GIPC_TOI_FILTER_RATIO","GIPC_TOI_REDUCED_SLACK","GIPC_TOI_TRIAL_INJECTIVITY",
            "GIPC_TOI_SAFE_INJECTIVITY","GIPC_TOI_REUSE_INITIAL_ASSEMBLY","GIPC_TOI_VOLUME_BOUND",
            "GIPC_TOI_STALL_WINDOW","GIPC_TOI_STALL_HOLD_DELTA","GIPC_TOI_EARLY_TERMINATION"})
            result["requested_environment"][name]=std::getenv(name)?Json(std::getenv(name)):Json(nullptr);
        result["policy_overrides"]=Json::array();
        if(robust_port)
        {
            for(const char* name:{"GIPC_TOI_WARM_CONTACTS","GIPC_TOI_WARM_LAMBDA","GIPC_TOI_WARM_GAMMA"})
                result["policy_overrides"].push_back({{"option",name},{"effective",true},{"reason","robust_policy_forces_retention"}});
            result["policy_overrides"].push_back({{"option","GIPC_TOI_ROBUST_VELOCITY_STOP"},{"effective",true},{"reason","robust_policy_forces_velocity_stop"}});
        }
        return result;
    }
};

// Shared by IPC and TOI call sites. These are resolved configuration values;
// observed Graph execution and preconditioner dispatch remain per-system data.
inline Json resolved_execution_options(const char* backend,double dt,double newton_tol,double pcg_tol)
{
    const bool cholesky=toi_flag_one("GIPC_MAS_CHOLESKY");
    const bool inverse64=toi_flag_one("GIPC_MAS_INVERSE64")&&!cholesky;
    const bool wide=toi_flag_one("GIPC_MAS_WIDE_APPLY")||inverse64||cholesky;
    Json result={{"schema_version",1},{"contact_backend",backend},{"dt",dt},
        {"ipc_newton_tol",newton_tol},{"pcg_rho_tol",pcg_tol},
        {"mas",{{"cholesky",cholesky},{"inverse64",inverse64},{"wide_apply",wide},
            {"restriction_mode",cholesky?(mas_warp_restrict()?"warp":"serial"):"inactive"},
            {"factor_action",cholesky?mas_factor_action_mode():"inactive"},
            {"requested_wide_apply",toi_flag_one("GIPC_MAS_WIDE_APPLY")},
            {"requested_inverse64",toi_flag_one("GIPC_MAS_INVERSE64")},
            {"wide_reason",cholesky?"required_by_cholesky":inverse64?"required_by_inverse64":"requested_or_default"},
            {"inverse64_overridden_by_cholesky",cholesky&&toi_flag_one("GIPC_MAS_INVERSE64")}}}};
    result["configured_pcg_execution"]=std::getenv("GIPC_PCG_EXECUTION")?Json(std::getenv("GIPC_PCG_EXECUTION")):Json("host");
    result["ipc_stopping"]=IpcOptions::read(newton_tol).json();
    result["requested_preconditioner"]=std::getenv("GIPC_PCG_PRECONDITIONER")?Json(std::getenv("GIPC_PCG_PRECONDITIONER")):Json(nullptr);
    result["execution_observed_per_linear_system"]=true;
    result["acceleration_features"]=Json::object();
    for(const char* name:{"GIPC_CCD_BVH_REFIT","GIPC_BATCHED_ENERGY","GIPC_ENERGY_REUSE",
        "GIPC_MAS_STATIC_TOPOLOGY"})
        result["acceleration_features"][name]=gipc_accel_feature(name);
    // This opt-in does not inherit GIPC_ACCEL_SUITE, and MAS cannot use it.
    result["requested_fused_diag_update"]=toi_flag_one("GIPC_PCG_FUSED_DIAG_UPDATE");
    result["fixed_mas_stage_study"]=toi_flag_one("GIPC_FIXED_MAS_STAGE_STUDY");
    result["legacy_restrict"]=legacy_ordered_restrict()?"ordered":"atomic";
    result["fixed_legacy_restrict_study"]=legacy_restrict_study();
    // Report candidates are separate opt-ins: configuration is not a claim
    // that a particular system can use the path. PCG/tree logs report usage.
    const auto& discrete=discrete_bvh_config();
    result["report_components"]={
        {"mas_fused_dot_requested",mas_fused_dot_requested()},
        {"fixed_mas_dot_study",mas_fused_dot_study_requested()},
        {"spmv_fused_quadratic_requested",spmv_fused_quadratic_requested()},
        {"fixed_spmv_quadratic_study",spmv_quadratic_study_requested()},
        {"discrete_bvh_refit",discrete.enabled},
        {"discrete_bvh_rebuild_interval",discrete.rebuild_interval},
        {"discrete_bvh_validate",discrete.validate},
        {"bvh_eligibility",false},
        {"bvh_eligibility_validate",false},
        {"bvh_eligibility_available",false},
        {"bounded_ccd",false},
        {"bounded_ccd_validate",false},
        {"bounded_ccd_available",false},
        {"bounded_ccd_scope","second_ipc_swept_query_only"},
        {"contact_pool",ipc_contact_pool_config().enabled},
        {"contact_pool_validate",ipc_contact_pool_config().validate},
        {"contact_pool_scope","ipc_line_search_discrete_query_only"},
        {"usage_observed_per_system_and_frame",true}};
    const bool cost_active=std::getenv("GIPC_COST_TRACE") && std::getenv("GIPC_COST_TRACE")[0];
    result["cost_observation"]={{"active",cost_active},
        {"gpu_events_requested",toi_flag_unless_zero("GIPC_COST_EVENTS")},
        {"gpu_events_effective",cost_active && toi_flag_unless_zero("GIPC_COST_EVENTS")},
        {"mode",cost_active?(toi_flag_unless_zero("GIPC_COST_EVENTS")?"cuda_events_nvtx_cpu":"nvtx_cpu_only"):"disabled"}};
    return result;
}
} // namespace gipc
