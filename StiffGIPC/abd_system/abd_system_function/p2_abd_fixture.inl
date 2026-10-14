int p2_abd_fixture(GIPC& owner,const char* output)
{
    device_TetraData tet;
    ABDSimData sim(owner,tet);ABDSystem system;
    system.parms.dt=.04;system.parms.kappa=5000;
    system.parms.motor_speed=.7;system.parms.motor_strength=3;
    Json rows=Json::array();
    for(int count:{1,3,513,0,2})
    {
        owner.abd_fem_count_info.abd_body_num=count;
        owner.abd_fem_count_info.abd_body_offset=1;
        owner.abd_fem_count_info.fem_body_num=1;
        std::vector<BodyBoundaryType> boundaries(count+1,BodyBoundaryType::Fixed);
        std::vector<Vector12> q(count),predicted(count),previous(count);
        std::vector<Float> volumes(count,.3);
        std::vector<ABDJacobiDyadicMass> masses(count);
        for(int i=0;i<count;++i)
        {
            boundaries[i+1]=i%3==0?BodyBoundaryType::Free:i%3==1?BodyBoundaryType::Fixed:BodyBoundaryType::Motor;
            q[i]=Vector12::Zero();q[i][3]=1+.001*(i+1);q[i][7]=1;q[i][11]=1;
            predicted[i]=q[i];predicted[i][0]=-.01;previous[i]=predicted[i];
            masses[i]=ABDJacobiDyadicMass(1,Vector3::Zero());
            for(int j=0;j<3;++j)masses[i]+=ABDJacobiDyadicMass(1,Vector3::Unit(j));
        }
        tet.body_id_to_boundary_type.copy_from(boundaries);
        sim.device.body_id_to_q.copy_from(q);sim.device.body_id_to_q_prev.copy_from(previous);
        sim.device.body_id_to_q_tilde.copy_from(predicted);sim.device.body_id_to_volume.copy_from(volumes);
        sim.device.body_id_to_abd_mass.copy_from(masses);
        const std::array<Float,2> original={system.cal_abd_kinetic_energy(sim),system.cal_abd_shape_energy(sim)};
        const auto pair=system.cal_abd_energies(sim);
        cudatool::DeviceBuffer<Float> into_batch(std::vector<Float>(11,-123));
        system.enqueue_abd_energies(sim,into_batch.data()+9);
        std::vector<Float> downloaded;into_batch.copy_to(downloaded);
        bool equal=std::memcmp(original.data(),pair.data(),sizeof(pair))==0
            &&std::memcmp(original.data(),downloaded.data()+9,sizeof(pair))==0;
        for(int j=0;j<9;++j)equal=equal&&downloaded[j]==-123;
        rows.push_back({{"bodies",count},{"kinetic",original[0]},{"shape",original[1]},{"bitwise_equal",equal}});
        if(!equal)throw std::runtime_error("P2 ABD fixture mismatch");
    }
    std::ofstream stream(output);if(!stream)throw std::runtime_error("Cannot write P2 ABD fixture");
    stream<<Json{{"passed",true},{"cases",rows}}.dump(2);
    return stream?0:1;
}
