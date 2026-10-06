#include "../StiffGIPC/solver/ipc_residual_controller.h"
#include "../StiffGIPC/solver/ipc_options.h"
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
int checks=0;
void require(bool value,const char* property) {
    ++checks;
    if(!value)throw std::runtime_error(property);
}
bool close(double a,double b) {
    return std::abs(a-b)<=128*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(a),std::abs(b)});
}
void environment(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value?value:"");
#else
    if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
void clear_options() {
    for(const char* name:{"GIPC_IPC_TERMINATION","GIPC_IPC_CUMULATIVE_TOL",
        "GIPC_IPC_RESIDUAL_REL_TOL","GIPC_IPC_RESIDUAL_FLOOR","GIPC_IPC_MIN_UPDATES",
        "GIPC_IPC_TERMINAL_AUDIT_FRAME","GIPC_IPC_RESIDUAL_SHADOW","GIPC_IPC_RESIDUAL_CPU_AUDIT"})
        environment(name,nullptr);
}
void reject_option(const char* name,const char* value) {
    clear_options();environment(name,value);
    bool threw=false;try{(void)gipc::IpcOptions::read(.01);}catch(const std::exception&){threw=true;}
    require(threw,"native invalid option not rejected");
}

void budget_properties() {
    // Check the published identity/inequalities over independently selected
    // zero/full/interior steps and rising/falling residuals, not a CPU solver.
    for(double w:{1.0,1.0/30}) {
        gipc::IpcBudget b;b.start(1,1e-30);
        for(int i=0;i<400;++i) {
            double alpha=(i%13)/12.0,residual=((37*i)%251)/100.0;
            b.accept(alpha);double r=b.observe(residual,w);
            require(b.valid&&!b.pending,"valid sequence lost budget validity");
            require(close(b.budget,b.beta+w*(b.unit-b.beta)),"weighted budget identity");
            require(b.budget+1e-13>=b.beta&&b.budget+1e-13>=w*r&&b.budget<=b.unit+1e-13,"budget inequalities");
            if(w==1)require(b.budget==b.unit,"weight one compatibility");
            if(alpha==1)require(b.beta==0&&close(b.unit,r)&&close(b.budget,w*r),"full step must clear old history");
        }
    }
    gipc::IpcBudget b;b.start(0,1e-30);b.accept(0);b.observe(0,1);
    require(b.valid&&b.beta==1&&!b.compensated(0,.001,.03),"zero step must not spend cumulative budget");
    for(bool active:{false,true}) {
        for(double a:{-1.,1.0001,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            gipc::IpcBudget x;if(active)x.start(1,1e-30);x.accept(a);x.start(1,1e-30);
            require(!x.valid&&!x.gated(0,.001,.03),"invalid alpha must be sticky before/after activation");
        }
        for(double r:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            gipc::IpcBudget x;if(active)x.start(1,1e-30);x.observe(r,1.0/30);x.start(1,1e-30);
            require(!x.valid&&!x.compensated(0,.001,.03),"invalid residual must be sticky before/after activation");
        }
    }
    gipc::IpcBudget duplicate;duplicate.start(1,1e-30);duplicate.accept(.5);duplicate.accept(.5);
    require(!duplicate.valid,"duplicate accept must invalidate");
    gipc::IpcBudget repeat;repeat.start(2,1e-30);repeat.start(200,1e-30);
    require(repeat.reference==2,"reference cannot reset");
    for(double w:{0.,-1.,1.01,std::numeric_limits<double>::quiet_NaN()}) {
        gipc::IpcBudget x;x.observe(1,w);x.start(1,1e-30);
        require(!x.valid,"invalid weight before activation must be sticky");
    }
}

void control_timing() {
    gipc::IpcResidualController c(.001,.03,1e-30,6,10);
    for(int accepted=0;accepted<5;++accepted) {
        auto o=c.observe(2,1,accepted,10,1);
        require(!c.budget().active&&!o.reference_frozen_now,"premature reference activation");
        c.accept(.5);
    }
    auto start=c.observe(2,1,5,10,1);
    require(start.reference_frozen_now&&c.reference_update()==5&&c.budget().reference==2,"freeze before sixth accepted update");
    require(c.budget().audits==0&&!c.budget().pending,"activation must not invent prior audits");
    c.accept(.5);
    require(c.budget().pending&&c.budget().beta==1&&c.budget().audits==0,"accept must defer residual audit");
    auto next=c.observe(.2,.1,6,20,1);
    require(next.had_pending&&next.audited&&next.audited_alpha==.5&&c.budget().audits==1,"next assembly audits final alpha once");
    require(c.budget().reference==2&&c.reference_epoch()==0&&c.objective_epoch()==1,"Kappa change must not reset reference");
    c.accept(1);
    auto animation=c.observe(.02,0,7,20,.99);
    require(animation.compensated_ready&&!animation.compensated_exit_allowed,"strict animation completion gate");
    auto finish=c.observe(.02,0,7,20,1);
    require(!finish.audited&&c.budget().audits==2,"repeated observation must not duplicate accepted audit");
    require(finish.compensated_exit_allowed,"valid completed animation exit");
    require(std::string(gipc::IpcResidualController::exit_reason(true,"compensated",finish))=="movement","movement exit priority");
    require(std::string(gipc::IpcResidualController::exit_reason(false,"compensated",finish))=="compensated","compensated selector");
    require(!gipc::IpcResidualController::exit_reason(false,"legacy",finish),"legacy must not use new exit");

    gipc::IpcResidualController h(.001,.03,1e-30,6,1);
    h.observe(1,0,5,1,1);h.accept(.9999);h.observe(3,0,6,1,1);
    h.accept(.5);auto veto=h.observe(.001,0,7,1,1);
    require(veto.gated_ready&&!veto.compensated_ready,"independent historical veto");
    h.accept(1);auto reset=h.observe(.002,0,8,1,1);
    require(reset.compensated_exit_allowed&&h.budget().beta==0&&close(h.budget().budget,reset.relative/30),"full step resets historical veto");

    gipc::IpcResidualController zero(.001,.03,1e-30,6,1);
    zero.observe(1,0,5,1,1);zero.accept(0);auto z=zero.observe(0,0,5,1,1);
    require(z.audited&&zero.budget().beta==1&&!z.minimum_updates_ready,"zero step is audited without increasing effective updates");
    auto nan=zero.observe(std::numeric_limits<double>::quiet_NaN(),0,5,1,1);
    require(!zero.budget().valid&&!nan.compensated_exit_allowed,"nonfinite free component must propagate");
}

void options() {
    clear_options();auto legacy=gipc::IpcOptions::read(.01);
    require(legacy.termination=="legacy"&&legacy.cumulative_tol==.01&&!legacy.residual_enabled(),"legacy defaults unchanged");
    environment("GIPC_IPC_TERMINATION","compensated");
    environment("GIPC_IPC_CUMULATIVE_TOL",".001");
    auto report=gipc::IpcOptions::read(.01);
    require(report.residual_enabled()&&report.residual_tol==.03&&report.min_updates==6,"report compatible options");
    for(const char* value:{"6x","6.0","-1"," 6","+6","0","10001"})reject_option("GIPC_IPC_MIN_UPDATES",value);
    for(const char* value:{"2x","-1","1.0"})reject_option("GIPC_IPC_TERMINAL_AUDIT_FRAME",value);
    for(const char* name:{"GIPC_IPC_RESIDUAL_SHADOW","GIPC_IPC_RESIDUAL_CPU_AUDIT"})
        for(const char* value:{"true","2","0x"})reject_option(name,value);
    reject_option("GIPC_IPC_RESIDUAL_CPU_AUDIT","1");
    reject_option("GIPC_IPC_TERMINAL_AUDIT_FRAME","1");
    reject_option("GIPC_IPC_CUMULATIVE_TOL","nan");
    reject_option("GIPC_IPC_RESIDUAL_FLOOR","0");
    clear_options();
}
}
int main() {
    try {
        budget_properties();control_timing();options();
        std::cout<<"{\"passed\":true,\"property_checks\":"<<checks
                 <<",\"audited_stress_steps\":800,\"controller_groups\":3}"<<std::endl;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
