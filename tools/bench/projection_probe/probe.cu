#include <native_balanced_gpu.cuh>
#include "shared_projection.cuh"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace gipc;
using Json=nlohmann::json;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
constexpr int calls=32;
const char* methods[]={"input_reference","input_fused","output_reference","output_naive","output_shared"};
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
void write_json(const fs::path& p,const Json& j){require(!fs::exists(p),"JSON exists");std::ofstream f(p);f<<j.dump(2);f.close();require(bool(f),"JSON write failed");}
std::string save(const fs::path& p,const std::vector<double>& a){require(!fs::exists(p),"Binary exists");
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(double));f.close();
    require(bool(f),"Binary write failed");return fs::absolute(p).generic_string();}
std::vector<double> load(const std::string& p,size_t count){std::vector<double> a(count);std::ifstream f(p,std::ios::binary);
    require(f.good(),"Missing binary input");f.read(reinterpret_cast<char*>(a.data()),count*sizeof(double));
    require(size_t(f.gcount())==count*sizeof(double)&&f.peek()==std::char_traits<char>::eof(),"Input byte count differs");
    for(double v:a)require(std::isfinite(v),"Nonfinite input");return a;}
struct Stream {cudaStream_t value=nullptr;Stream(){balanced_cuda(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value)cudaStreamDestroy(value);}};
struct Events {cudaEvent_t begin=nullptr,end=nullptr;Events(){balanced_cuda(cudaEventCreate(&begin));
    try{balanced_cuda(cudaEventCreate(&end));}catch(...){cudaEventDestroy(begin);throw;}}
    ~Events(){if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}};
struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;size_t nodes=0;Json types=Json::object();double setup_ms=0;
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}};
struct Data {std::string name;int n;std::vector<double> Z,W,r,u,a,c;};
struct Device {
    const int n;Data data;Stream stream;BalancedBlas blas;Events events;
    BalancedBuffer<double> Z,W,r,a,c,out,work_c;
    std::vector<double> init_out,init_c;Eigen::VectorXd input_expected,output_expected;
    std::map<int,std::unique_ptr<Graph>> graphs;
    explicit Device(Data value):n(value.n),data(std::move(value)),Z(size_t(n)*8),W(size_t(n)*8),r(n),a(8),c(8),
        out(size_t(n)*calls),work_c(size_t(8)*calls),init_out(size_t(n)*calls),init_c(size_t(8)*calls) {
        require(n>0&&n<=100000,"Probe size outside bounded range");
        balanced_blas(cublasSetStream(blas.handle,stream.value));
        auto upload=[&](double* d,const std::vector<double>& h){balanced_cuda(cudaMemcpyAsync(d,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice,stream.value));};
        upload(Z.value,data.Z);upload(W.value,data.W);upload(r.value,data.r);upload(a.value,data.a);upload(c.value,data.c);
        for(int k=0;k<calls;++k){std::copy(data.u.begin(),data.u.end(),init_out.begin()+size_t(k)*n);
            std::copy(data.c.begin(),data.c.end(),init_c.begin()+size_t(k)*8);}
        balanced_cuda(cudaStreamSynchronize(stream.value));
        Eigen::Map<const Eigen::MatrixXd> hZ(data.Z.data(),n,8),hW(data.W.data(),n,8);
        Eigen::Map<const Eigen::VectorXd> hr(data.r.data(),n),hu(data.u.data(),n),ha(data.a.data(),8),hc(data.c.data(),8);
        input_expected=hr-hW*ha;output_expected=hu+hZ*(ha-hc);
        require(input_expected.allFinite()&&output_expected.allFinite(),"CPU formula nonfinite");
    }
    double reset(){const auto t=Clock::now();
        balanced_cuda(cudaMemcpyAsync(out.value,init_out.data(),init_out.size()*sizeof(double),cudaMemcpyHostToDevice,stream.value));
        balanced_cuda(cudaMemcpyAsync(work_c.value,init_c.data(),init_c.size()*sizeof(double),cudaMemcpyHostToDevice,stream.value));
        balanced_cuda(cudaStreamSynchronize(stream.value));return elapsed(t);}
    void op(int method,int k){const double one=1,minus=-1;double* target=out.value+size_t(k)*n;
        switch(method){
        case 0:
            balanced_cuda(cudaMemcpyAsync(target,r.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,stream.value));
            balanced_blas(cublasDgemv(blas.handle,CUBLAS_OP_N,n,8,&minus,W.value,n,a.value,1,&one,target,1));break;
        case 1:balanced_input_projection<<<(n+255)/256,256,0,stream.value>>>(n,r.value,W.value,a.value,target);break;
        case 2:
            balanced_difference<<<1,8,0,stream.value>>>(a.value,work_c.value+size_t(k)*8);
            balanced_blas(cublasDgemv(blas.handle,CUBLAS_OP_N,n,8,&one,Z.value,n,work_c.value+size_t(k)*8,1,&one,target,1));break;
        case 3:balanced_output_projection<<<(n+255)/256,256,0,stream.value>>>(n,Z.value,a.value,c.value,target);break;
        case 4:shared_output_projection<<<(n+255)/256,256,0,stream.value>>>(n,Z.value,a.value,c.value,target);break;
        default:throw std::runtime_error("Invalid method");}
        balanced_cuda(cudaGetLastError());}
    std::vector<double> read(){std::vector<double> values(size_t(n)*calls);
        balanced_cuda(cudaMemcpyAsync(values.data(),out.value,values.size()*sizeof(double),cudaMemcpyDeviceToHost,stream.value));
        balanced_cuda(cudaStreamSynchronize(stream.value));return values;}
    Json check(const std::vector<double>& values,int method,double tolerance){
        const auto& expected=method<2?input_expected:output_expected;double worst=0,max_abs=0;
        for(int k=0;k<calls;++k){double error=0;for(int i=0;i<n;++i){const double v=values[size_t(k)*n+i];
            require(std::isfinite(v),"Nonfinite projection output");const double d=v-expected[i];error+=d*d;max_abs=std::max(max_abs,std::abs(d));}
            worst=std::max(worst,std::sqrt(error/std::max(expected.squaredNorm(),1e-300)));}
        require(std::isfinite(worst)&&worst<=tolerance,"Projection CPU formula check failed");
        std::vector<double> coef(size_t(8)*calls);
        balanced_cuda(cudaMemcpyAsync(coef.data(),work_c.value,coef.size()*sizeof(double),cudaMemcpyDeviceToHost,stream.value));
        balanced_cuda(cudaStreamSynchronize(stream.value));
        for(int k=0;k<calls;++k)for(int j=0;j<8;++j){const double expected_c=method==2?data.a[j]-data.c[j]:data.c[j];
            require(coef[size_t(k)*8+j]==expected_c,"Coefficient work bank invalid");}
        return {{"max_bank_relative_l2",worst},{"max_absolute_error",max_abs},{"all_32_outputs_checked",true}};
    }
    bool unchanged(){auto equal=[&](const double* d,const std::vector<double>& original){std::vector<double> copy(original.size());
        balanced_cuda(cudaMemcpyAsync(copy.data(),d,copy.size()*sizeof(double),cudaMemcpyDeviceToHost,stream.value));
        balanced_cuda(cudaStreamSynchronize(stream.value));return std::memcmp(copy.data(),original.data(),copy.size()*sizeof(double))==0;};
        return equal(Z.value,data.Z)&&equal(W.value,data.W)&&equal(r.value,data.r)&&equal(a.value,data.a)&&equal(c.value,data.c);}
    void capture(int method){auto g=std::make_unique<Graph>();const auto t=Clock::now();reset();
        balanced_cuda(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
        for(int k=0;k<calls;++k)op(method,k);
        balanced_cuda(cudaStreamEndCapture(stream.value,&g->graph));
        balanced_cuda(cudaGraphInstantiate(&g->exec,g->graph,0));
        balanced_cuda(cudaGraphGetNodes(g->graph,nullptr,&g->nodes));std::vector<cudaGraphNode_t> nodes(g->nodes);
        balanced_cuda(cudaGraphGetNodes(g->graph,nodes.data(),&g->nodes));
        for(auto node:nodes){cudaGraphNodeType type;balanced_cuda(cudaGraphNodeGetType(node,&type));
            const std::string key=std::to_string(int(type));g->types[key]=g->types.value(key,0)+1;}
        require(g->nodes>=calls,"Missing graph operations");if(method==1||method==3||method==4)require(g->nodes==calls,"Unexpected fused graph nodes");
        g->setup_ms=elapsed(t);graphs[method]=std::move(g);}
    Json timed(int method,bool graph){const auto t=Clock::now();balanced_cuda(cudaEventRecord(events.begin,stream.value));
        if(graph)balanced_cuda(cudaGraphLaunch(graphs.at(method)->exec,stream.value));
        else for(int k=0;k<calls;++k)op(method,k);
        balanced_cuda(cudaEventRecord(events.end,stream.value));balanced_cuda(cudaEventSynchronize(events.end));
        const double host=elapsed(t);float event=0;balanced_cuda(cudaEventElapsedTime(&event,events.begin,events.end));
        return {{"host_ms",host},{"event_ms",event},{"calls",calls},{"graph",graph},{"method",methods[method]}};}
};
Json prepare(Device& d,const fs::path& out,double tolerance){Json rows=Json::array();
    for(int method=0;method<5;++method){d.reset();for(int k=0;k<calls;++k)d.op(method,k);
        balanced_cuda(cudaStreamSynchronize(d.stream.value));auto values=d.read();Json host=d.check(values,method,tolerance);
        host.update({{"case",d.data.name},{"n",d.n},{"method",methods[method]},{"graph",false},
            {"output_file",save(out/(d.data.name+"_"+methods[method]+"_host_warm.bin"),values)}});rows.push_back(host);
        d.capture(method);d.reset();balanced_cuda(cudaGraphLaunch(d.graphs.at(method)->exec,d.stream.value));
        balanced_cuda(cudaStreamSynchronize(d.stream.value));values=d.read();Json graph=d.check(values,method,tolerance);
        graph.update({{"case",d.data.name},{"n",d.n},{"method",methods[method]},{"graph",true},
            {"nodes",d.graphs.at(method)->nodes},{"node_types",d.graphs.at(method)->types},
            {"graph_setup_host_ms",d.graphs.at(method)->setup_ms},
            {"output_file",save(out/(d.data.name+"_"+methods[method]+"_graph_warm.bin"),values)}});rows.push_back(graph);}
    require(d.unchanged(),"Read-only input changed");return rows;}
Data fixture(int n){Data d;d.name="contract_"+std::to_string(n);d.n=n;d.r.resize(n);d.u.resize(n);d.Z.resize(size_t(n)*8);d.W=d.Z;d.a.resize(8);d.c.resize(8);
    for(int j=0;j<8;++j){d.a[j]=.13*(j-3);d.c[j]=.07*(4-j);}
    for(int i=0;i<n;++i){d.r[i]=std::sin(.17*i)+.3;d.u[i]=std::cos(.12*i)-.2;
        for(int j=0;j<8;++j){d.W[i+size_t(j)*n]=std::cos(.013*(i+1)*(j+1));d.Z[i+size_t(j)*n]=std::sin(.021*(i+1)*(j+2));}}return d;}
Json device_info(){cudaDeviceProp p{};balanced_cuda(cudaGetDeviceProperties(&p,0));std::ostringstream uuid;
    for(unsigned char c:p.uuid.bytes)uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
    return {{"name",p.name},{"uuid_hex",uuid.str()},{"major",p.major},{"minor",p.minor}};}
int main(int argc,char** argv){try{
    require(argc==3||argc==4,"Usage: contracts out | run manifest out");balanced_cuda(cudaSetDevice(0));
    const std::string mode=argv[1];const fs::path out=argv[argc-1];require(fs::is_directory(out),"Output directory missing");
    Json result={{"device",device_info()},{"calls_per_batch",calls},{"warm",Json::array()},{"samples",Json::array()},{"passed",false}};
    if(mode=="contracts"){
        require(argc==3,"Contracts arguments");for(int n:{1,31,32,33,255,256,257,1025}){Device d(fixture(n));
            const auto t=Clock::now();auto rows=prepare(d,out,2e-12);for(auto& r:rows)result["warm"].push_back(r);
            result["contract_setup_ms"][std::to_string(n)]=elapsed(t);}
        result["passed"]=true;write_json(out/"contracts.json",result);std::cout<<"80 host/graph projection contracts passed\n";return 0;
    }
    require(mode=="run"&&argc==4,"Invalid action");std::ifstream input(argv[2]);Json manifest;input>>manifest;
    std::vector<std::unique_ptr<Device>> cases;
    for(const auto& row:manifest.at("cases")){Data d;d.name=row.at("name");d.n=row.at("n");require(d.n>0&&d.n<=100000,"Input size");
        const auto& f=row.at("files");d.Z=load(f.at("Z"),size_t(d.n)*8);d.W=load(f.at("W"),size_t(d.n)*8);
        d.r=load(f.at("r"),d.n);d.u=load(f.at("u"),d.n);d.a=load(f.at("a"),8);d.c=load(f.at("c"),8);
        auto ptr=std::make_unique<Device>(std::move(d));const auto setup=Clock::now();auto rows=prepare(*ptr,out,1e-10);
        for(auto& warm:rows)result["warm"].push_back(warm);result["case_setup_ms"][ptr->data.name]=elapsed(setup);cases.push_back(std::move(ptr));}
    require(cases.size()==4,"Expected four fixed-input probes");
    const int orders[6][3]={{2,3,4},{4,3,2},{3,2,4},{4,2,3},{2,4,3},{3,4,2}};
    for(int round=0;round<3;++round){std::vector<int> ids={0,1,2,3};if(round==1)std::reverse(ids.begin(),ids.end());
        if(round==2)std::rotate(ids.begin(),ids.begin()+2,ids.end());
        for(int id:ids){Device& d=*cases[id];
            for(int rep=0;rep<6;++rep)for(int group=0;group<2;++group)for(int mode_order=0;mode_order<2;++mode_order){
                const bool graph=(round+mode_order)%2!=0;const int count=group?3:2;
                for(int order=0;order<count;++order){const int method=group?orders[rep][order]:(rep+order)%2;
                    const double reset_ms=d.reset();Json row=d.timed(method,graph);const auto check_start=Clock::now();
                    auto values=d.read();row.update(d.check(values,method,1e-10));const double validation_ms=elapsed(check_start);
                    row.update({{"case",d.data.name},{"n",d.n},{"round",round+1},{"repeat",rep+1},{"group",group},
                        {"mode_order",mode_order+1},{"order",order+1},{"reset_host_ms",reset_ms},{"validation_host_ms",validation_ms}});
                    const auto export_start=Clock::now();if(rep==0){std::vector<double> first(values.begin(),values.begin()+d.n),last(values.end()-d.n,values.end());
                        const std::string tag=d.data.name+"_r"+std::to_string(round+1)+"_"+methods[method]+(graph?"_graph":"_host");
                        row["first_file"]=save(out/(tag+"_first.bin"),first);row["last_file"]=save(out/(tag+"_last.bin"),last);}
                    row["export_host_ms"]=elapsed(export_start);result["samples"].push_back(row);
                }
            }
            require(d.unchanged(),"Timed probe changed read-only inputs");
        }
        std::cout<<"projection round "<<round+1<<" complete\n"<<std::flush;
    }
    result["read_only_inputs_unchanged"]=true;result["passed"]=true;write_json(out/"measurements.json",result);
    std::cout<<"720 paired projection samples passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
