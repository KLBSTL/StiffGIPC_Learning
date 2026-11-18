// P2 observation only. Source sets are captured before CCD/line search and
// committed only when the corresponding physical update is accepted.
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
namespace gipc {
inline bool reuse_observation_enabled() {
    const char* p=std::getenv("GIPC_REUSE_OBSERVE_DIR");return p&&p[0];
}
template<int N> __device__ void reuse_observe_hessian(double2* output,int index,const double (&h)[N][N]) {
    if(!output)return;
    double square=0;bool finite=true;
    for(int i=0;i<N;++i)for(int j=0;j<N;++j){finite=finite&&isfinite(h[i][j]);square+=h[i][j]*h[i][j];}
    output[index]=make_double2(finite&&isfinite(square)?(square>0?1:2):-1,sqrt(square));
}
struct ReuseObservation {
    using Key=std::array<int,10>; // term, kind, role-preserving vertices, object IDs
    struct Item {int multiplicity=0;double hessian_norm=0;std::array<double,4> geometry{};};
    std::array<cudatool::DeviceBuffer<double2>,4> hessian;
    std::map<Key,Item> previous,pending;
    Json row;int source_solve=-1,source_accepted=0;bool staged=false;
    std::ofstream events;
    void emit(const char* event) {
        if(!reuse_observation_enabled())return;
        if(!events.is_open()) {
            const auto dir=std::filesystem::path(std::getenv("GIPC_REUSE_OBSERVE_DIR"));
            std::filesystem::create_directories(dir);const auto file=dir/"solves.jsonl";
            if(std::filesystem::exists(file))throw std::runtime_error("Refusing to overwrite P2 observations");
            events.open(file);
        }
        auto copy=row;copy["event"]=event;events<<copy.dump()<<'\n';events.flush();
        if(!events)throw std::runtime_error("P2 event write failed");
    }
    void begin_frame(){previous.clear();pending.clear();source_solve=-1;source_accepted=0;staged=false;}
    void prepare(int a,int b,int c,int d) {
        if(!reuse_observation_enabled())return;
        const int counts[]={a,b,c,d};
        for(int i=0;i<4;++i){hessian[i].resize(counts[i]);
            if(counts[i])CUDA_SAFE_CALL(cudaMemset(hessian[i].data(),0,counts[i]*sizeof(double2)));}
    }
    double2* buffer(int i){return reuse_observation_enabled()?hessian[i].data():nullptr;}
    template<class T> static std::vector<T> read(const T* device,size_t n) {
        std::vector<T> result(n);if(n)CUDA_SAFE_CALL(cudaMemcpy(result.data(),device,n*sizeof(T),cudaMemcpyDeviceToHost));return result;
    }
    static Key key(int term,int4 raw,const std::vector<int>& bodies) {
        int kind;std::array<int,4> v{raw.x,raw.y,raw.z,raw.w};
        if(term>=2){kind=8;v={raw.x,0,-1,-1};}
        else if(raw.x>=0){kind=raw.w>=0?1:2;if(raw.w<0)v[3]=-raw.w-1;
            if(v[0]>v[1])std::swap(v[0],v[1]);if(v[2]>v[3])std::swap(v[2],v[3]);
            if(std::make_pair(v[0],v[1])>std::make_pair(v[2],v[3])){std::swap(v[0],v[2]);std::swap(v[1],v[3]);}}
        else {
            v[0]=-raw.x-1;
            if(raw.z<0 && raw.y<0){kind=3;v[1]=-raw.y-1;v[2]=-raw.z-1;v[3]=-raw.w-1;
                if(v[0]>v[1]){std::swap(v[0],v[1]);std::swap(v[2],v[3]);}}
            else if(raw.z<0){kind=4;v[2]=v[3]=-1;if(v[0]>v[1])std::swap(v[0],v[1]);}
            else if(raw.w<0 && raw.y<0){kind=5;v[1]=-raw.y-1;v[3]=-raw.w-1;if(v[1]>v[2])std::swap(v[1],v[2]);}
            else if(raw.w<0){kind=6;v[3]=-1;if(v[1]>v[2])std::swap(v[1],v[2]);}
            else {kind=7;std::sort(v.begin()+1,v.end());}
        }
        Key out{term,kind,v[0],v[1],v[2],v[3],-1,-1,-1,-1};
        for(int i=0;i<4;++i)if(v[i]>=0 && !(term>=2&&i==1)) {
            if(size_t(v[i])>=bodies.size())throw std::runtime_error("Contact key vertex out of range");out[6+i]=bodies[v[i]];
        }
        return out;
    }
    static std::array<double,4> geometry(const Key& k,const std::vector<double3>& x,const double3& gn,double go) {
        auto v=[&](int i){const auto p=x[k[i+2]];return Eigen::Vector3d(p.x,p.y,p.z);};
        Eigen::Vector3d delta,normal;double distance=0;
        if(k[0]>=2){normal=Eigen::Vector3d(gn.x,gn.y,gn.z);distance=normal.dot(v(0))-go;}
        else if(k[1]==3||k[1]==4){delta=v(0)-v(1);distance=delta.norm();normal=distance>0?Eigen::Vector3d(delta/distance):Eigen::Vector3d::Zero();}
        else if(k[1]==5||k[1]==6){const Eigen::Vector3d edge=v(2)-v(1);delta=v(0)-v(1);
            if(edge.squaredNorm()>0)delta-=edge*(edge.dot(delta)/edge.squaredNorm());
            distance=delta.norm();normal=distance>0?Eigen::Vector3d(delta/distance):Eigen::Vector3d::Zero();}
        else {normal=k[1]==7?Eigen::Vector3d((v(2)-v(1)).cross(v(3)-v(1))):Eigen::Vector3d((v(1)-v(0)).cross(v(3)-v(2)));
            const double norm=normal.norm();if(norm>0){normal/=norm;distance=normal.dot(k[1]==7?v(0)-v(1):v(2)-v(0));}
            else normal.setZero();}
        if(distance<0){distance=-distance;normal=-normal;}
        return {distance,normal[0],normal[1],normal[2]};
    }
    void stage(int frame,int solve,int accepted,double kappa,double dt,const double3* positions,int vertices,
        const int* body_ids,const int* boundary,int abd_vertices,const BodyBoundaryType* abd_boundary,int bodies_count,
        const int4* pairs,int np,const int4* friction,int nf,const uint32_t* ground,int ng,
        const uint32_t* friction_ground,int nfg,const double3* normal,const double* offset) {
        if(!reuse_observation_enabled())return;
        const auto started=std::chrono::steady_clock::now();pending.clear();staged=true;
        const auto x=read(positions,vertices);const auto objects=read(body_ids,vertices),bc=read(boundary,vertices);
        const auto ab=read(abd_boundary,bodies_count);const auto gn=read(normal,1)[0];const double go=read(offset,1)[0];
        std::vector<bool> free(vertices);for(int i=0;i<vertices;++i)
            free[i]=i<abd_vertices?(objects[i]>=0 && objects[i]<bodies_count && ab[objects[i]]!=BodyBoundaryType::Fixed):bc[i]==0;
        const auto p=read(pairs,np),f=read(friction,nf);const auto g=read(ground,ng),fg=read(friction_ground,nfg);
        const int counts[]={np,nf,ng,nfg};Json raw=Json::array(),hist=Json::object();int skipped=0,zeros=0,constrained=0;
        // Raw identities are a bounded collection check, not a trajectory log.
        const bool dump=solve<=2&&std::getenv("GIPC_REUSE_OBSERVE_RAW")&&std::string(std::getenv("GIPC_REUSE_OBSERVE_RAW"))=="1";
        for(int term=0;term<4;++term) {
            if(hessian[term].size()!=size_t(counts[term]))throw std::runtime_error("P2 assembly observation size mismatch");
            const auto weights=read(hessian[term].data(),counts[term]);
            for(int i=0;i<counts[term];++i) {
                const int4 record=term==0?p[i]:term==1?f[i]:make_int4(int(term==2?g[i]:fg[i]),0,-1,-1);
                const auto k=key(term,record,objects);const auto w=weights[i];
                if(w.x<0||!std::isfinite(w.y))throw std::runtime_error("Nonfinite observed contact Hessian");
                bool active_free=false;for(int j=0;j<4;++j)if(k[j+2]>=0&&!(term>=2&&j==1))active_free=active_free||free[k[j+2]];
                if(dump)raw.push_back({{"term",term},{"raw",{record.x,record.y,record.z,record.w}},
                    {"written",w.x!=0},{"nonzero",w.x==1},{"has_free_support",active_free},{"hessian_norm",w.y},{"key",k}});
                if(w.x==0){++skipped;continue;}if(w.x==2){++zeros;continue;}if(!active_free){++constrained;continue;}
                auto& item=pending[k];++item.multiplicity;item.hessian_norm+=w.y;item.geometry=geometry(k,x,gn,go);
                const auto name=std::to_string(term)+":"+std::to_string(k[1]);hist[name]=hist.value(name,0)+1;
            }
        }
        size_t common=0;int multiplicity_changes=0;double distance_change=0,normal_change=0,curvature_change=0;
        for(const auto& [k,v]:pending)if(auto it=previous.find(k);it!=previous.end()) {
            ++common;const auto& old=it->second;multiplicity_changes+=v.multiplicity!=old.multiplicity;
            distance_change=std::max(distance_change,std::abs(v.geometry[0]-old.geometry[0]));double normal2=0;
            for(int j=1;j<4;++j){const double a=v.geometry[j]-old.geometry[j];normal2+=a*a;}
            normal_change=std::max(normal_change,std::sqrt(normal2));
            curvature_change=std::max(curvature_change,std::abs(v.hessian_norm-old.hessian_norm)/std::max({v.hessian_norm,old.hessian_norm,1e-300}));
        }
        auto fingerprint=[](const auto& values){uint64_t h=14695981039346656037ull;for(const auto& [key,item]:values)
            for(int value:key)for(int j=0;j<4;++j){h^=(uint32_t(value)>>(8*j))&255;h*=1099511628211ull;}return h;};
        const size_t united=previous.size()+pending.size()-common;
        row={{"schema","pcg-observation-1"},{"frame",frame},{"solve",solve},{"accepted_before",accepted},
            {"source_solve",source_solve},{"source_accepted",source_accepted},{"available_history",std::min(accepted,4)},
            {"kappa",kappa},{"dt",dt},{"contact_unique",pending.size()},{"source_contact_unique",previous.size()},
            {"contact_added",pending.size()-common},{"contact_removed",previous.size()-common},
            {"contact_jaccard",united?double(united-common)/united:0.},{"contact_hash",fingerprint(pending)},
            {"source_contact_hash",fingerprint(previous)},{"multiplicity_changes",multiplicity_changes},
            {"common_distance_abs_max_m",distance_change},{"common_normal_delta_max",normal_change},
            {"common_hessian_norm_relative_max",curvature_change},{"term_kind_counts",hist},
            {"dcd_records",np},{"friction_records",nf},{"ground_records",ng},{"ground_friction_records",nfg},
            {"skipped_before_hessian_write",skipped},{"zero_hessian_terms",zeros},{"fixed_only_terms",constrained},
            {"contact_scope","nonzero assembled contact Hessian terms with free physical support; not swept CCD pairs"},
            {"observation_host_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()}};
        if(dump){row["raw_records"]=raw;row["canonical_multiset"]=Json::array();for(const auto& [k,v]:pending)
            row["canonical_multiset"].push_back({{"key",k},{"multiplicity",v.multiplicity},{"hessian_norm",v.hessian_norm},{"geometry",v.geometry}});}
        emit("solve_begin");
        row.erase("raw_records");row.erase("canonical_multiset");
        Statistics::instance().at_current_frame()["newton"].back()["reuse_observation"]=row;
    }
    void solved(){if(!staged)return;const auto& n=Statistics::instance().at_current_frame()["newton"].back();
        row["pcg"]=n["pcg"];row["mas_structure"]=n.value("mas_diagnostic",Json::object());emit("solve_end");}
    void accept(double alpha){if(!staged)return;row["alpha"]=alpha;row["accepted"]=std::isfinite(alpha)&&alpha>0;
        if(row["accepted"].get<bool>()){previous=std::move(pending);source_solve=row["solve"].get<int>();source_accepted=row["accepted_before"].get<int>()+1;}
        emit("accepted_update");staged=false;}
};
inline ReuseObservation& reuse_observation(){static thread_local ReuseObservation state;return state;}
} // namespace gipc
