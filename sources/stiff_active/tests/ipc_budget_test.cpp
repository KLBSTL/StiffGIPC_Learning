#include "../StiffGIPC/solver/ipc_budget.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
void require(bool x){if(!x)throw std::runtime_error("IPC budget regression failed");}
int main(){
    using gipc::IpcBudget;
    IpcBudget b;b.start(10,1e-30);b.accept(.5);double r=b.observe(4,1.0/3);
    require(b.valid&&r==.4&&b.beta==.5&&b.budget==.5);
    b.accept(1);r=b.observe(.1,1.0/3);
    require(b.valid&&b.beta==0&&std::abs(b.budget-r/3)<1e-15&&b.gated(r,.01,.03)&&b.compensated(r,.01,.03));
    IpcBudget history;history.start(1,1e-30);history.accept(.99);r=history.observe(.01,1.0/3);
    history.accept(.5);r=history.observe(.001,1.0/3);
    require(history.gated(r,.01,.03)&&history.compensated(r,.01,.03));
    IpcBudget veto;veto.start(1,1e-30);veto.accept(.99);veto.observe(3,1.0/3);
    veto.accept(.5);r=veto.observe(.001,1.0/3);
    require(veto.valid&&veto.gated(r,.01,.03)&&!veto.compensated(r,.01,.03));
    IpcBudget zero;zero.start(0,1e-30);zero.accept(0);r=zero.observe(0,1);require(zero.valid&&zero.beta==1&&!zero.gated(r,.01,.03));
    for(double a:{-1.0,1.01,std::numeric_limits<double>::infinity()}){IpcBudget x;x.start(1,1e-30);x.accept(a);require(!x.valid);}
    IpcBudget duplicate;duplicate.start(1,1e-30);duplicate.accept(.5);duplicate.accept(.5);require(!duplicate.valid);
    IpcBudget inf;inf.start(1,1e-30);inf.accept(.5);inf.observe(std::numeric_limits<double>::infinity(),1);require(!inf.valid);
    std::cout<<"IPC budget invariants, full/zero step, historical veto and invalid inputs passed\n";
}
