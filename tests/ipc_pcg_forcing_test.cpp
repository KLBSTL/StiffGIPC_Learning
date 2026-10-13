#undef NDEBUG
#include <solver/ipc_pcg_forcing.h>
#include <cassert>
#include <limits>
#include <string>
int main() {
    gipc::IpcPcgForcingOptions options;options.mode="adaptive";
    gipc::IpcPcgForcingController controller(options);
    auto initial=controller.observe(100,0,2,1);assert(initial.eta==.01&&initial.epoch==1);
    auto progress=controller.observe(10,1,2,1);
    assert(std::abs(progress.eta-.9*std::pow(.1,1.5))<1e-15);
    assert(std::abs(progress.eta*progress.eta-.00081)<1e-15);
    auto stalled=controller.observe(11,2,2,1);assert(stalled.eta==.01);
    auto reset=controller.observe(20,3,4,1);assert(reset.eta==.01&&reset.epoch==2);
    auto near=controller.observe(.5,4,4,1);assert(near.eta==.01);
    auto target=controller.observe(.3,5,4,.9);assert(target.eta==.01&&target.epoch==3);
    bool duplicate=false;
    try{controller.observe(.2,5,4,.9);}catch(const std::runtime_error&){duplicate=true;}
    assert(duplicate);
    bool invalid=false;
    try{controller.observe(std::numeric_limits<double>::quiet_NaN(),6,4,.9);}
    catch(const std::runtime_error&){invalid=true;}
    assert(invalid);
    options.mode="fixed";options.fixed_eta=.03;
    gipc::IpcPcgForcingController fixed(options);
    assert(fixed.observe(1,0,1,1).eta==.03);
    options.mode="monitor";gipc::IpcPcgForcingController monitor(options);
    assert(monitor.observe(10,0,1,1).eta==.01);
    assert(monitor.observe(9,1,1,1).eta==.01);
    options.mode="adaptive";gipc::IpcPcgForcingController cap(options);
    cap.observe(10,0,1,1);assert(cap.observe(9,1,1,1).eta==.1);
    return 0;
}
