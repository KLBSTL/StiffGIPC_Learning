//
// gl_main.cpp
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GL/glew.h"
#include "GL/freeglut.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <map>
// #include "GIPC.cuh"
#include "device_launch_parameters.h"
#include "mlbvh.cuh"
#include <stdio.h>
#include "load_mesh.h"
#include "cuda_tools/cuda_tools.h"
#include <queue>
//#include "timer.h"
#include "femEnergy.cuh"
#include "gpu_eigen_libs.cuh"
#include "fem_parameters.h"
#include "gipc_path.h"
#include <gipc/type_define.h>
#include <filesystem>
#include <gipc/statistics.h>
#include <gipc/utils/json.h>
#include <numeric>
#include <set>
#include <algorithm>
#include <cmath>
#include <gipc/utils/simple_scene_importer.h>
#include <Eigen/Geometry>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <GIPC.cuh>
#include <abd_system/abd_system.h>
#include <cstdlib>

auto             assets_dir = std::string{gipc::assets_dir()};
std::string      metis_dir  = assets_dir + "sorted_mesh/";
double           collision_detection_buff_scale = 1;
double           motion_rate                    = 1;
double           linear_system_buff_scale       = 1.0;
mesh_obj         obj;
lbvh_f           bvh_f;
lbvh_e           bvh_e;
GIPC             ipc;
device_TetraData d_tetMesh;
tetrahedra_obj   tetMesh;
std::vector<Node>     nodes;
std::vector<AABB>     bvs;
std::vector<std::string>   obj_pathes;
int              initPath = 0;
int   step      = 0;
int   frameId   = 0;
int   surfNumId = 0;
float xRot      = 0.0f;
float yRot      = 0.f;
float xTrans    = 0;
float yTrans    = 0;
float zTrans    = 0;
int   ox;
int   oy;
int   buttonState;
float xRotLength    = 0.0f;
float yRotLength    = 0.0f;
float window_width  = 1000;
float window_height = 1000;
int   s_dimention   = 3;
bool  saveSurface   = false;
bool  change        = false;
bool  screenshot    = false;

bool isSetShader = false;

bool drawbvh     = false;
bool drawSurface = true;

bool stop = true;

double3 center;
double3 Ssize;

GLuint PN_vbo_;
GLuint VAO;
GLuint color_vbo_;
//GLuint color_vao_;
GLuint normal_vbo_;
//GLuint normal_vao_;
GLuint v;
GLuint f;
GLuint shaderProgram;

int            clothFaceOffset = 0;
int            bodyVertOffset  = 0;
double         global_offset   = 1.0;
int            selected_scene  = 1;
std::string    benchmark_cloth_case;
bool           experimental_graph_spmv = false;
bool           experimental_device_scalar_pcg = false;
bool           experimental_pcg_graph_segment = false;
bool           experimental_pcg_graph_cache = false;
bool           experimental_pcg_graph_update = false;
bool           experimental_pcg_graph_tail_cache = false;
bool           experimental_pcg_conditional_while = false;
bool           experimental_pcg_conditional_mas = false;
bool           experimental_pcg_conditional_cache = false;
bool           experimental_pcg_fused_dot_tail = false;
bool           experimental_pcg_fused_continue = false;
bool           experimental_intersection_scratch = false;
bool           experimental_mas_static_topology = false;
bool           experimental_mas_contact_topology = false;
bool           experimental_mas_fused_clear = false;
bool           experimental_reuse_newton_events = false;
bool           experimental_batched_energy = false;
bool           experimental_energy_reuse = false;
bool           experimental_energy_reuse_audit = false;
bool           experimental_ccd_bvh_refit = false;
bool           experimental_ccd_bvh_refit_audit = false;
bool           experimental_defect_shadow = false;
bool           experimental_defect_export_vectors = false;
bool           experimental_adaptive_pcg = false;
bool           experimental_verified_fixed_pcg = false;
bool           experimental_pcg_legacy_stop = false;
int            experimental_b_mode = 0;
double         experimental_eps_r = 0.0;
double         experimental_b_guard_multiplier = 1.0;
bool           experimental_b_strict = true;
bool           experimental_b_gradient_split = false;
bool           experimental_b_gradient_audit = false;
bool           experimental_b_quiet_trace = false;
bool           experimental_b_terminal_fast_stop = false;
bool           experimental_b_fused_defect_reduction = false;
int*           benchmark_collision_body_ids = nullptr;
std::string    benchmark_scene_manifest_path;
gipc::Json     benchmark_scene_manifest;
std::string    benchmark_object_manifest_path;
gipc::Json     benchmark_object_manifest;
extern int     benchmark_scene_id;
extern std::string benchmark_output_dir;
extern int     benchmark_newton_cap_hits;
extern int     benchmark_export_path_frame;
std::vector<std::string> files;
std::vector<int>    file_vert_offsets;
std::vector<int>    file_tet_offsets;

void Init_CUDA()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }
}

#pragma pack(push, 1)
typedef struct
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} mBITMAPFILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} mBITMAPINFOHEADER;
#pragma pack(pop)

bool WriteBitmapFile(int width, int height, const std::string& file_name, unsigned char* bitmapData)
{
    mBITMAPFILEHEADER bitmapFileHeader;
    memset(&bitmapFileHeader, 0, sizeof(mBITMAPFILEHEADER));
    bitmapFileHeader.bfSize = sizeof(mBITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;  //BM
    bitmapFileHeader.bfOffBits = sizeof(mBITMAPFILEHEADER) + sizeof(mBITMAPINFOHEADER);

    mBITMAPINFOHEADER bitmapInfoHeader;
    memset(&bitmapInfoHeader, 0, sizeof(mBITMAPINFOHEADER));
    bitmapInfoHeader.biSize        = sizeof(mBITMAPINFOHEADER);
    bitmapInfoHeader.biWidth       = width;
    bitmapInfoHeader.biHeight      = height;
    bitmapInfoHeader.biPlanes      = 1;
    bitmapInfoHeader.biBitCount    = 24;
    bitmapInfoHeader.biCompression = 0L;
    bitmapInfoHeader.biSizeImage   = width * std::abs(height) * 3;

    //////////////////////////////////////////////////////////////////////////
    FILE*         filePtr;
    unsigned char tempRGB;
    int           imageIdx;

    for(imageIdx = 0; imageIdx < (int)bitmapInfoHeader.biSizeImage; imageIdx += 3)
    {
        tempRGB                  = bitmapData[imageIdx];
        bitmapData[imageIdx]     = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }

    filePtr = fopen(file_name.c_str(), "wb");
    if(NULL == filePtr)
    {
        return false;
    }

    fwrite(&bitmapFileHeader, sizeof(mBITMAPFILEHEADER), 1, filePtr);

    fwrite(&bitmapInfoHeader, sizeof(mBITMAPINFOHEADER), 1, filePtr);

    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

    fclose(filePtr);
    return true;
}

void SaveScreenShot(int width, int height, const std::string& file_name)
{
    int   data_len    = height * width * 3;  // bytes
    void* screen_data = malloc(data_len);
    memset(screen_data, 0, data_len);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_data);
    WriteBitmapFile(width, height, file_name + ".bmp", (unsigned char*)screen_data);
    free(screen_data);
}

void saveSurfaceMesh(const std::string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId++) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    std::ofstream    outSurf(file_path);

    std::map<int, int> meshToSurf;
    outSurf << "s 1" << std::endl;
    for(int i = 0; i < tetMesh.surfVerts.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[tetMesh.surfVerts[i]];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.surface.size(); i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf << "f " << meshToSurf[tri.x] + 1 << " " << meshToSurf[tri.y] + 1
                << " " << meshToSurf[tri.z] + 1 << std::endl;
    }
    outSurf.close();
}


void saveTets(const std::string& path)
{
    int tetIdoffset = 0;
    for(int ii = 0; ii < 4096; ii++)
    {
        //tetMesh.output_tetrahedraMesh
        std::stringstream ss;
        ss << path;
        ss << ii;  // / 10;
        //if (surfNumId % 10 != 0) return;
        ss << ".msh";
        std::string file_path = ss.str();
        std::ofstream    outmsh1(file_path);

        std::map<int, int> meshToSurf;
        //outSurf << "s 1" << std::endl;
        outmsh1 << "$Nodes\n";
        outmsh1 << file_vert_offsets[ii + 1] - file_vert_offsets[ii] << std::endl;
        for(int i = 0; i < file_vert_offsets[ii + 1] - file_vert_offsets[ii]; i++)
        {
            const auto& pos = tetMesh.vertexes[i + file_vert_offsets[ii]];
            outmsh1 << i + 1 << " " << pos.x << " " << pos.y << " " << pos.z << std::endl;
            meshToSurf[i + file_vert_offsets[ii]] = i;
        }
        outmsh1 << "$Elements\n";
        outmsh1 << file_tet_offsets[ii + 1] << std::endl;

        for(int i = 0; i < file_tet_offsets[ii + 1]; i++)
        {
            int tetId = i + tetIdoffset;
            outmsh1 << i + 1 << " 4 0 " << meshToSurf[tetMesh.tetrahedras[tetId].x] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].y] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].z] + 1 << " "
                    << meshToSurf[tetMesh.tetrahedras[tetId].w] + 1 << std::endl;
        }
        tetIdoffset += file_tet_offsets[ii + 1];
        outmsh1.close();
    }
}

void draw_box2D(float ox, float oy, float width, float height)
{
    glLineWidth(2.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox + width, oy, 0);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox, oy + height, 0);

    glVertex3f(ox + width, oy, 0);
    glVertex3f(ox + width, oy + height, 0);

    glVertex3f(ox + width, oy + height, 0);
    glVertex3f(ox, oy + height, 0);

    glEnd();
}

void draw_box3D(float ox, float oy, float oz, float width, float height, float length, int boxType = 0)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.1f);
    if(boxType == 1)
    {
        glLineWidth(1.5f);
        glColor3f(0.8f, 0.8f, 0.8f);
    }
    glBegin(GL_LINES);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);

    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);

    glEnd();
}

void draw_lines(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);
    int numbers = 20;
    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox + width * i / numbers, oy, 0);
        glVertex3f(ox + width * i / numbers, oy + height, 0);
    }

    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy + height * i / numbers, 0);
        glVertex3f(ox + width, oy + height * i / numbers, 0);
    }

    glEnd();


    glLineWidth(1.5f);
    glColor3f(0.8f, 0.8f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(ox + width / 2, oy, 0);
    glVertex3f(ox + width / 2, oy + height, 0);

    glVertex3f(ox, oy + height / 2, 0);
    glVertex3f(ox + width, oy + height / 2, 0);

    glEnd();
}

void draw_mesh3D()
{
    glEnable(GL_DEPTH_TEST);
    glLineWidth(1.5f);
    glColor3f(0.9f, 0.1f, 0.1f);
    const std::vector<uint3>& surf = tetMesh.surface;  //obj.faces;
    glBegin(GL_TRIANGLES);


    for(int j = 0; j < tetMesh.surface.size(); j++)
    {
        glVertex3f((tetMesh.vertexes[surf[j].x].x),
                   (tetMesh.vertexes[surf[j].x].y),
                   (tetMesh.vertexes[surf[j].x].z));
        glVertex3f((tetMesh.vertexes[surf[j].y].x),
                   (tetMesh.vertexes[surf[j].y].y),
                   (tetMesh.vertexes[surf[j].y].z));
        glVertex3f((tetMesh.vertexes[surf[j].z].x),
                   (tetMesh.vertexes[surf[j].z].y),
                   (tetMesh.vertexes[surf[j].z].z));
    }
    glEnd();

    glColor3f(0.9f, 0.9f, 0.9f);
    //glDisable(GL_DEPTH_TEST);
    glLineWidth(0.1f);
    glBegin(GL_LINES);

    for(int j = 0; j < tetMesh.surfEdges.size(); j++)
    {
        //if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 965) || (tetMesh.surfEdges[j].x == 965 && tetMesh.surfEdges[j].y == 870)) {
        //    glColor3f(0.9f, 0.1f, 0.1f);
        //    glLineWidth(3.4f);
        //}
        //else if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 905) || (tetMesh.surfEdges[j].x == 905 && tetMesh.surfEdges[j].y == 870)) {
        //    glColor3f(0.9f, 0.9f, 0.1f);
        //    glLineWidth(3.4f);
        //}

        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].x].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].z));
        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].y].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].z));

        glColor3f(0.9f, 0.9f, 0.9f);
        glLineWidth(0.1f);
    }
    glEnd();

    //glColor3f(0.99f, 0.1f, 0.1f);
    ////glDisable(GL_DEPTH_TEST);
    //glPointSize(8);
    //glBegin(GL_POINTS);
    //glVertex3f((tetMesh.vertexes[2189].x), (tetMesh.vertexes[2189].y), (tetMesh.vertexes[2189].z));
    //glColor3f(0.99f, 0.99f, 0.1f);
    //glVertex3f((tetMesh.vertexes[870].x), (tetMesh.vertexes[870].y), (tetMesh.vertexes[870].z));
    //glVertex3f((tetMesh.vertexes[905].x), (tetMesh.vertexes[905].y), (tetMesh.vertexes[905].z));
    //glVertex3f((tetMesh.vertexes[965].x), (tetMesh.vertexes[965].y), (tetMesh.vertexes[965].z));
    //glEnd();
}

void draw_bvh()
{
    int num = (bvs.size() + 1) / 2;
    for(int j = 0; j < bvs.size(); j++)
    {
        int   i = j;
        float ox, oy, oz, bwidth, bheight, blength;
        ox      = (bvs[i].lower.x);
        oy      = (bvs[i].lower.y);
        oz      = (bvs[i].lower.z);
        bwidth  = (bvs[i].upper.x - bvs[i].lower.x);
        bheight = (bvs[i].upper.y - bvs[i].lower.y);
        blength = (bvs[i].upper.z - bvs[i].lower.z);
        draw_box3D(ox, oy, oz, bwidth, bheight, blength);
    }
}

int            counttt = 0;
std::vector<float3> getRenderGeometry(int& number)
{

    std::vector<double3> meshNormal(tetMesh.vertexNum, make_double3(0, 0, 0));
    number = tetMesh.surface.size();  //meshTemp.surfaceRender.size();
    std::vector<float3> pos_normal_color(3 * number * 3);

    for(int i = 0; i < number; i++)
    {
        //int tetId = meshTemp.surfaceRender[i][3];
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        double3 vt0 = tetMesh.vertexes[v0];  // Vector3d(meshTemp.vertexes[v0][0], meshTemp.vertexes[v0][1], meshTemp.vertexes[v0][2]);
        double3 vt1 = tetMesh.vertexes[v1];  // Vector3d(meshTemp.vertexes[v1][0], meshTemp.vertexes[v1][1], meshTemp.vertexes[v1][2]);
        double3 vt2 = tetMesh.vertexes[v2];  // Vector3d(meshTemp.vertexes[v2][0], meshTemp.vertexes[v2][1], meshTemp.vertexes[v2][2]);
        double3 vec1 = __GEIGEN__::__minus(vt1, vt0);  //vt1 - vt0;
        double3 vec2 = __GEIGEN__::__minus(vt2, vt0);
        double3 normal =
            __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(vec1, vec2));  //vec1.cross(vec2).normalized();

        pos_normal_color[i * 9]     = make_float3(vt0.x, vt0.y, vt0.z);
        pos_normal_color[i * 9 + 3] = make_float3(vt1.x, vt1.y, vt1.z);
        pos_normal_color[i * 9 + 6] = make_float3(vt2.x, vt2.y, vt2.z);

        pos_normal_color[i * 9 + 2] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 5] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 8] = make_float3(0.6875f, 0.51953f, 0.38671f);
        //}


        meshNormal[v0] = __GEIGEN__::__add(meshNormal[v0], normal);  //normal;
        meshNormal[v1] = __GEIGEN__::__add(meshNormal[v1], normal);
        meshNormal[v2] = __GEIGEN__::__add(meshNormal[v2], normal);
    }
    for(int i = 0; i < number; i++)

    {
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        //meshNormal[v0].normalize(); meshNormal[v1].normalize(); meshNormal[v2].normalize();
        pos_normal_color[i * 9 + 1] =
            make_float3(meshNormal[v0].x, meshNormal[v0].y, meshNormal[v0].z);
        pos_normal_color[i * 9 + 4] =
            make_float3(meshNormal[v1].x, meshNormal[v1].y, meshNormal[v1].z);
        pos_normal_color[i * 9 + 7] =
            make_float3(meshNormal[v2].x, meshNormal[v2].y, meshNormal[v2].z);
    }

    return pos_normal_color;
}


void draw_Scene3D()
{
    //face.mesh3Ds[0] = mesh3d;
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(xTrans, yTrans, zTrans);
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    //draw_box3D(-2, -1, -2, 4, 4, 4, 1);
    if(drawSurface)
    {
        draw_mesh3D();
    }
    if(drawbvh)
    {
        draw_bvh();
    }

    glPopMatrix();


    glutSwapBuffers();
    //glFlush();
}
double mfsum                   = 0;
double total_time              = 0;
int    total_cg_iterations     = 0;
int    total_newton_iterations = 0;
int    start                   = -1;

void saveScreenPic(const std::string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << step;
    std::string file_path = ss.str();

    SaveScreenShot(window_width, window_height, file_path);
}

void initFEM(tetrahedra_obj& mesh)
{

    double massSum   = 0;
    double volumeSum = 0;
    //float  angleX = FEM::PI / 4, angleY = -FEM::PI / 4, angleZ = FEM::PI / 2;
    //__GEIGEN__::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
    //__GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
    //__GEIGEN__::__set_Mat_val(
    //    rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));


    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));

    ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                    / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));

    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;

    printf("ipc.shearStiff: %f\n", ipc.shearStiff);


    for(int i = 0; i < mesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);  //calculateDms3D_double(mesh.vertexes, mesh.tetrahedras[i], 0);

        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);

        double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

        mesh.masses[mesh.tetrahedras[i].x] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * ipc.density / 4;

        massSum += vlm * ipc.density;
        volumeSum += vlm;
        mesh.DM_inverse.push_back(DM_inverse);
        mesh.volum.push_back(vlm);


        double lengthRateLame =
            mesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double volumeRateLame = mesh.vert_youngth_modules[i] * ipc.PoissonRate
                                / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        double lengthRate = 4 * lengthRateLame / 3;
        double volumeRate = volumeRateLame + 5 * lengthRateLame / 6;

        mesh.lengthRate.push_back(lengthRate);
        mesh.volumeRate.push_back(volumeRate);
    }

    for(int i = 0; i < mesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d DM;
        __calculateDm2D_double(mesh.vertexes.data(), mesh.triangles[i], DM);

        __GEIGEN__::Matrix2x2d DM_inverse;
        __GEIGEN__::__Inverse2x2(DM, DM_inverse);

        double area = calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
        area *= ipc.clothThickness;
        mesh.area.push_back(area);


        mesh.masses[mesh.triangles[i].x] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].y] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].z] += ipc.clothDensity * area / 3;

        massSum += area * ipc.clothDensity;
        volumeSum += area;
        mesh.tri_DM_inverse.push_back(DM_inverse);
    }

    mesh.meanMass = massSum / mesh.vertexNum;
    printf("meanMass: %f\n", mesh.meanMass);
    mesh.meanVolum = volumeSum / mesh.vertexNum;
}

void DefaultSettings()
{
    // global settings
    ipc.density        = 1e3;
    ipc.PoissonRate    = 0.49;
    //ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
    //                     / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    //ipc.lengthRate        = 4 * ipc.lengthRateLame / 3;
    //ipc.volumeRate        = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.frictionRate      = 0.4;
    ipc.gd_frictionRate   = 0.4;
    ipc.clothThickness    = 1e-3;
    ipc.clothYoungModulus = 1e6;
    ipc.bendYoungModulus  = 1e5;
    //ipc.stretchStiff      = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.shearStiff        = ipc.stretchStiff * 0.3;
    ipc.clothDensity      = 2e2;
    ipc.strainRate        = 100;
    ipc.softMotionRate    = 1e0;
    ipc.bendStiff         = 3e-4;
    ipc.Newton_solver_threshold = 1e-2;
    ipc.pcg_threshold           = 1e-4;
    ipc.IPC_dt                  = 1e-2;
    ipc.relative_dhat           = 1e-3;
    //ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
    //                / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    //ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;
}
//int  meshids = 0;
void LoadSettings()
{
    bool successfulRead = false;

    //read file
    std::ifstream infile;


    std::string DEFAULT_CONFIG_FILE = std::string{gipc::assets_dir()} + "scene/parameterSetting.txt";


    infile.open(DEFAULT_CONFIG_FILE, std::ifstream::in);
    if(successfulRead = infile.is_open())
    {
        int  tempEnum;
        char ignoreToken[256];

        // global settings:
        infile >> ignoreToken >> ipc.density;
        infile >> ignoreToken >> ipc.PoissonRate;
        infile >> ignoreToken >> ipc.frictionRate;
        infile >> ignoreToken >> ipc.gd_frictionRate;
        infile >> ignoreToken >> ipc.clothThickness;
        infile >> ignoreToken >> ipc.clothYoungModulus;
        infile >> ignoreToken >> ipc.bendYoungModulus;
        //infile >> ignoreToken >> ipc.shearStiff;
        infile >> ignoreToken >> ipc.clothDensity;
        infile >> ignoreToken >> ipc.strainRate;
        infile >> ignoreToken >> ipc.softMotionRate;
        //infile >> ignoreToken >> ipc.bendStiff;
        infile >> ignoreToken >> collision_detection_buff_scale;
        infile >> ignoreToken >> motion_rate;
        infile >> ignoreToken >> ipc.IPC_dt;
        infile >> ignoreToken >> ipc.pcg_threshold;
        infile >> ignoreToken >> ipc.Newton_solver_threshold;
        infile >> ignoreToken >> ipc.relative_dhat;
        //infile >> ignoreToken >> meshids;



        //ipc.shearStiff =
        infile.close();
    }

    if(!successfulRead)
    {
        std::cerr << "Waning: failed loading settings, set to defaults." << std::endl;
        DefaultSettings();
    }
}

void set_case1()
{
    double                    dist       = 0.2;
    int                       count      = 4;
    int                       count_Y    = 4;
    double                    fem_height = -0.8;
    double                    abd_height = -0.6;
    gipc::SimpleSceneImporter importer;

    linear_system_buff_scale = 1.0;

    double Youngth_Modulus = 1e4;
    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -abd_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::ABD,
                                       transform,
                                       1e5,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -fem_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::FEM,
                                       transform,
                                       Youngth_Modulus,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }
}


void set_case2()
{
    ipc.pcg_data.P_type = 1;
    linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer;
    auto load_object = [&](size_t index,
                           const char* name,
                           int dimension,
                           gipc::BodyType body_type,
                           const char* default_mesh,
                           double default_scale,
                           const Eigen::Vector3d& default_translation,
                           double default_young) {
        std::string mesh = assets_dir + default_mesh;
        double scale = default_scale;
        Eigen::Vector3d translation = default_translation;
        double young = default_young;
        if(!benchmark_object_manifest_path.empty())
        {
            const auto& object = benchmark_object_manifest.at("objects").at(index);
            if(object.at("name") != name || object.at("dimension") != dimension
               || object.at("body_type") !=
                      (body_type == gipc::BodyType::ABD ? "ABD" : "FEM"))
                throw std::runtime_error("Object manifest scene1 object identity mismatch");
            mesh = (std::filesystem::path(assets_dir)
                    / object.at("stiff_mesh").get<std::string>()).string();
            scale = object.at("source_scale").get<double>();
            const auto& shift = object.at("world_translation_m");
            translation = Eigen::Vector3d(shift.at(0).get<double>(),
                                          shift.at(1).get<double>(),
                                          shift.at(2).get<double>());
            if(dimension == 3)
                young = object.at("tet_young_pa").get<double>();
        }
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
        transform.block<3, 1>(0, 3) = translation;
        importer.load_geometry(tetMesh,
                               dimension,
                               body_type,
                               transform,
                               young,
                               mesh,
                               ipc.pcg_data.P_type);
    };
    load_object(0, "abd_bunny", 3, gipc::BodyType::ABD,
                "tetMesh/bunny2.msh", 0.2, Eigen::Vector3d(0, 0.5, 0), 1e4);
    load_object(1, "fem_bunny", 3, gipc::BodyType::FEM,
                "tetMesh/bunny2.msh", 0.2, Eigen::Vector3d(0, -0.65, 0), 1e4);
    load_object(2, "cloth", 2, gipc::BodyType::FEM,
                "triMesh/cloth_high.obj", 1.0, Eigen::Vector3d::Zero(), 1e4);
}

void set_case3()
{

    gipc::SimpleSceneImporter importer{assets_dir + "scene/json/wrecking-ball-simple.json",
                                       assets_dir + "tetMesh/wrecking-ball-mesh/",
                                       gipc::BodyType::ABD};
    linear_system_buff_scale = 1.0;
    importer.import_scene(tetMesh);
}

void set_case4()
{
    ipc.pcg_data.P_type = 0;
    linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.6;
    double3                   position_offset = make_double3(0, 1.0, 0);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(Eigen::Vector3d{0, 1.0, 0});
    t.scale(scale);
    t.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));
    Eigen::Matrix4d transform = t.matrix();

    std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tetMesh.maxTConer.y;
    double       min_x            = tetMesh.minTConer.x;
    double       max_x            = tetMesh.maxTConer.x;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].y > max_y - eps
           && (tetMesh.vertexes[i].x < min_x + eps || tetMesh.vertexes[i].x > max_x - eps))
        {
            tetMesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "fixed vertex num: " << fixed_vertex_num << std::endl;
}

void set_case5()
{
    ipc.pcg_data.P_type = 1;
    linear_system_buff_scale = 2.0;
    gipc::SimpleSceneImporter importer;
    double                    scale = 1.0;
    Eigen::Vector3d           position_offset{0, 0, 0};

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(position_offset);
    t.scale(scale);
    Eigen::Matrix4d transform = t.matrix();

    std::string mesh_path       = assets_dir + "tetMesh/high_mat.msh";
    double Youngth_Modulus = 1e4;
    ipc.PoissonRate        = 0.48;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           Youngth_Modulus,
                           mesh_path,
                           ipc.pcg_data.P_type);

    // no gravity
    for(int i = 0; i < tetMesh.vertexes.size(); i++)
    {
        tetMesh.apply_gravity[i] = 0;
    }

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -0.5 + eps || tetMesh.vertexes[i].x > 0.5 - eps)
        {
            tetMesh.targetIndex.push_back(i);
            tetMesh.targetPos.push_back(tetMesh.vertexes[i]);
        }
    }
    tetMesh.softNum = tetMesh.targetIndex.size();
    std::cout << "soft constraint num: " << tetMesh.softNum << std::endl;
    ipc.softMotionRate = 1;

    const double angular_vel = 3.14159265358979323846/5;
    d_tetMesh.update_soft_constraint_functor =
        [angular_vel](double3 vertex, int step_id, double ipc_dt) -> double3
    {
        double3 rotated_vertex = vertex;
        if(vertex.x < 0)
        {
            // rotate along x axis clockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(angular_vel * ipc_dt)
                                  - vertex.z * std::sin(angular_vel * ipc_dt),
                              vertex.y * std::sin(angular_vel * ipc_dt)
                                  + vertex.z * std::cos(angular_vel * ipc_dt)};
        }
        if(vertex.x > 0)
        {
            // rotate along x axis counterclockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(-angular_vel * ipc_dt)
                                  - vertex.z * std::sin(-angular_vel * ipc_dt),
                              vertex.y * std::sin(-angular_vel * ipc_dt)
                                  + vertex.z * std::cos(-angular_vel * ipc_dt)};
        }
        return rotated_vertex;
    };
}

Eigen::Matrix4d benchmark_matrix16(const gipc::Json& values)
{
    Eigen::Matrix4d matrix;
    for(int row = 0; row < 4; ++row)
        for(int column = 0; column < 4; ++column)
            matrix(row, column) = values.at(row * 4 + column).get<double>();
    return matrix;
}

void set_case6()
{
    linear_system_buff_scale = 2.0;
    ipc.pcg_data.P_type = 1;
    if(!benchmark_object_manifest_path.empty())
    {
        const auto& objects = benchmark_object_manifest.at("objects");
        for(int group = 0; group < 2; ++group)
        {
            const auto& object = objects.at(group);
            const std::string name = group == 0 ? "stiff_boxes" : "soft_boxes";
            if(object.at("name") != name || object.at("body_type") != "ABD"
               || object.at("material_model") != "AffineBody"
               || object.at("fixed_mode") != "none"
               || object.at("instance_transforms_f64x16").size() != 960)
                throw std::runtime_error("Scene 5 box object identity mismatch");
            const auto source = benchmark_matrix16(object.at("source_transform_f64x16"));
            const auto mesh = (std::filesystem::path(assets_dir)
                / object.at("stiff_mesh").get<std::string>()).string();
            for(const auto& instance : object.at("instance_transforms_f64x16"))
                tetMesh.load_tetrahedraMesh(mesh, benchmark_matrix16(instance) * source,
                    object.at("young_pa").get<double>(), gipc::BodyType::ABD);
        }
        const auto& cloth = objects.at(2);
        if(cloth.at("name") != "cloth" || cloth.at("body_type") != "FEM"
           || cloth.at("fixed_mode") != "edge_x"
           || cloth.at("material_model") != "BaraffWitkinShell")
            throw std::runtime_error("Scene 5 cloth object identity mismatch");
        gipc::SimpleSceneImporter importer;
        const auto source = benchmark_matrix16(cloth.at("source_transform_f64x16"));
        const auto world = benchmark_matrix16(cloth.at("instance_transforms_f64x16").at(0)) * source;
        importer.load_geometry(tetMesh, 2, gipc::BodyType::FEM, world,
            1e4, (std::filesystem::path(assets_dir)
                 / cloth.at("stiff_mesh").get<std::string>()).string(), ipc.pcg_data.P_type);
        const double eps = 1e-4;
        for(int i = 0; i < tetMesh.vertexNum; ++i)
            if(tetMesh.vertexes[i].x < -1.5 + eps || tetMesh.vertexes[i].x > 1.5 - eps)
                tetMesh.boundaryTypies[i] = 1;
        ipc.relative_dhat = 1e-3;
        ipc.strainRate = 1e6;
        return;
    }
    double scale      = 0.3;
    double dist       = scale / 2;
    int    count      = 8;
    int    count_Y    = 15;
    double fem_height = global_offset + 1 - 0.8;
    double abd_height = fem_height - dist;


    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;


                double3 position_offset =
                    double3{-pos.x(), -abd_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            1e6,
                                            gipc::BodyType::ABD);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    double3{-pos.x(), -fem_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            5e4,
                                            gipc::BodyType::ABD);
            }
        }
    }
    
    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.scale(1.5);
    t.translate(Eigen::Vector3d(0, 0.35, 0));
    std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           t.matrix(),
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -1.5 + eps || tetMesh.vertexes[i].x > 1.5 - eps)
        {
            tetMesh.boundaryTypies[i] = 1;
        }
    }

    ipc.relative_dhat = 1e-3;
    ipc.strainRate    = 1e6;
}

void set_paper_animal_well()
{
    ipc.pcg_data.P_type = 0;
    ipc.PoissonRate = 0.3;
    linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer;
    auto load_object = [&](size_t index,
                           const char* name,
                           int dimension,
                           const char* default_mesh,
                           double default_young) {
        std::string mesh = assets_dir + default_mesh;
        double scale = 1.0;
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        double young = default_young;
        if(!benchmark_object_manifest_path.empty())
        {
            const auto& object = benchmark_object_manifest.at("objects").at(index);
            if(object.at("name") != name || object.at("body_type") != "FEM"
               || object.at("dimension") != dimension)
                throw std::runtime_error("Figure 4 object manifest identity mismatch");
            mesh = (std::filesystem::path(assets_dir)
                    / object.at("stiff_mesh").get<std::string>()).string();
            scale = object.at("source_scale").get<double>();
            const auto& shift = object.at("world_translation_m");
            translation = Eigen::Vector3d(shift.at(0).get<double>(),
                                          shift.at(1).get<double>(),
                                          shift.at(2).get<double>());
            if(dimension == 3)
                young = object.at("tet_young_pa").get<double>();
        }
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
        transform.block<3, 1>(0, 3) = translation;
        importer.load_geometry(tetMesh,
                               dimension,
                               gipc::BodyType::FEM,
                               transform,
                               young,
                               mesh,
                               ipc.pcg_data.P_type);
    };
    load_object(0, "animal", 3,
                "../barrier-free/source/assets/sim_data/tetmesh/animal_well.msh",
                5e5);
    const int pool_start = tetMesh.vertexNum;
    load_object(1, "pool", 2,
                "../barrier-free/source/assets/sim_data/trimesh/pool1.obj",
                1e6);
    for(int i = pool_start; i < tetMesh.vertexNum; ++i)
        tetMesh.boundaryTypies[i] = 1;
    // Some mesh loaders retain the larger individual body bbox rather than the
    // union. Use the actual initialized world vertices for an absolute 1 mm d_hat.
    if(tetMesh.vertexes.empty())
        throw std::runtime_error("Figure 4 scene has no vertices");
    double3 low = tetMesh.vertexes.front();
    double3 high = low;
    for(const auto& p : tetMesh.vertexes)
    {
        low.x = std::min(low.x, p.x);
        low.y = std::min(low.y, p.y);
        low.z = std::min(low.z, p.z);
        high.x = std::max(high.x, p.x);
        high.y = std::max(high.y, p.y);
        high.z = std::max(high.z, p.z);
    }
    tetMesh.minConer = low;
    tetMesh.maxConer = high;
    const double dx = high.x - low.x;
    const double dy = high.y - low.y;
    const double dz = high.z - low.z;
    ipc.relative_dhat = 0.001 / std::sqrt(dx * dx + dy * dy + dz * dz);
}
void setMAS_partition()
{
    tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());
    int index = 0;
    for(int i = 0; i < tetMesh.partId.size(); i++)
    {
        tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = i;
        index++;
        if(i <= tetMesh.partId.size() - 2)
        {
            if(tetMesh.partId[i + 1] != tetMesh.partId[i])
            {
                index = 0;
            }
        }
    }
    index = 0;
    for(int i = 0; i < tetMesh.partId_map_real.size(); i++)
    {

        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = i;
            index++;
        }
    }
}

#include "benchmark_cloth_scenes.inc"

// Apply the frozen scene's physical scalars after scene-specific defaults and
// before FEM masses, material matrices, and GPU buffers are constructed.
void apply_benchmark_scene_manifest()
{
    if(benchmark_scene_manifest_path.empty())
        return;
    const auto& fields = benchmark_scene_manifest.at("effective_scalar_fields");
    if(ipc.pcg_data.P_type != fields.at("preconditioner_type").get<int>())
        throw std::runtime_error("Scene manifest preconditioner differs from loaded mesh partition");
#ifdef USE_FRICTION
    constexpr bool friction_compiled = true;
#else
    constexpr bool friction_compiled = false;
#endif
    if(friction_compiled != fields.at("friction_compiled").get<bool>())
        throw std::runtime_error("Scene manifest friction build differs from executable");
    if(fields.at("cloth_stretch_model") != "BaraffWitkinStrainLimiting"
       || fields.at("cloth_bending_model") != "QuadraticBendingQ")
        throw std::runtime_error("Unsupported scene manifest cloth model");

    ipc.IPC_dt = fields.at("dt").get<double>();
    ipc.density = fields.at("density").get<double>();
    ipc.PoissonRate = fields.at("poisson_ratio").get<double>();
    ipc.clothDensity = fields.at("cloth_density").get<double>();
    ipc.clothThickness = fields.at("cloth_thickness").get<double>();
    ipc.clothYoungModulus = fields.at("cloth_young").get<double>();
    ipc.bendYoungModulus = fields.at("bending_young").get<double>();
    ipc.strainRate = fields.at("strain_rate").get<double>();
    ipc.frictionRate = fields.at("friction_rate").get<double>();
    ipc.gd_frictionRate = fields.at("ground_friction_rate").get<double>();
    ipc.Newton_solver_threshold =
        fields.at("newton_direction_threshold_coefficient").get<double>();
    ipc.pcg_threshold = fields.at("pcg_requested_threshold").get<double>();
    ipc.relative_dhat = fields.at("relative_dhat").get<double>();
    if(!(ipc.IPC_dt > 0 && ipc.density > 0 && ipc.clothDensity > 0
         && ipc.clothThickness > 0 && ipc.relative_dhat > 0
         && ipc.pcg_threshold > 0 && ipc.Newton_solver_threshold > 0))
        throw std::runtime_error("Invalid non-positive scene manifest physical scalar");
}

void initScene(bool initialize_only = false)
{
    std::filesystem::exists(metis_dir) || std::filesystem::create_directory(metis_dir);
    ipc.pcg_data.P_type = 1;

    int scene_no = selected_scene;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!ABD must be loaded before FEM!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    switch(scene_no)
    {
        case 0:  // box pipe
            set_case1();
            break;
        case 1:  // soft-rigid-cloth coupling
            set_case2();
            break;
        case 2:  //wrecking ball case
            set_case3();
            break;
        case 3:  //fixed cloth
            set_case4();
            break;
        case 4:  //twisting mat
            set_case5();
            break;
        case 5:  //box pipe large scale and cloth
            set_case6();
            break;
        case 6:  // paper animal well geometry, implicit five-wall container
            set_paper_animal_well();
            break;
        case 7:  // frozen cloth-v1 scenes, selected with --cloth-case
            set_case8();
            break;
    }


    apply_benchmark_scene_manifest();
    setMAS_partition();


    tetMesh.getSurface();

    initFEM(tetMesh);
    if(initialize_only)
    {
        if(benchmark_scene_manifest_path.empty())
            throw std::runtime_error("Initialization-only requires frozen scalar preflight");
        const auto& fields = benchmark_scene_manifest.at("effective_scalar_fields");
        ipc.bboxDiagSize2 = fields.at("bbox_diagonal_squared").get<double>();
        ipc.dHat = fields.at("dhat_squared").get<double>();
        return;
    }
    //device_TetraData d_tetMesh;
    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                tetMesh.tri_edges.size(),
                                tetMesh.abd_fem_count_info.total_body_num());

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses,
                              tetMesh.masses.data(),
                              tetMesh.vertexNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.apply_gravity,
                              tetMesh.apply_gravity.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.lengthRate,
                              tetMesh.lengthRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volumeRate,
                              tetMesh.volumeRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum,
                              tetMesh.volum.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tetrahedras,
                              tetMesh.tetrahedras.data(),
                              tetMesh.tetrahedraNum * sizeof(uint4),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.DmInverses,
                              tetMesh.DM_inverse.data(),
                              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.BoundaryType,
                              tetMesh.boundaryTypies.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities,
                              tetMesh.velocities.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetIndex,
                              tetMesh.targetIndex.data(),
                              tetMesh.softNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                              tetMesh.targetPos.data(),
                              tetMesh.softNum * sizeof(double3),
                              cudaMemcpyHostToDevice));

    d_tetMesh.host_target_indices  = tetMesh.targetIndex;
    d_tetMesh.host_target_vertices = tetMesh.targetPos;

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triDmInverses,
                              tetMesh.tri_DM_inverse.data(),
                              tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.area,
                              tetMesh.area.data(),
                              tetMesh.triangleNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triangles,
                              tetMesh.triangles.data(),
                              tetMesh.triangleNum * sizeof(uint3),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edges,
                              tetMesh.tri_edges.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edge_adj_vertex,
                              tetMesh.tri_edges_adj_points.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.body_id_to_boundary_type,
                              tetMesh.body_id_to_is_fixed.data(),
                              tetMesh.body_id_to_is_fixed.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.point_id_to_body_id,
                              tetMesh.point_id_to_body_id.data(),
                              tetMesh.point_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tet_id_to_body_id,
                              tetMesh.tet_id_to_body_id.data(),
                              tetMesh.tet_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));


    printf("stretchStiff:  %f,  shearStiff:   %f\n", ipc.stretchStiff, ipc.shearStiff);

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = tetMesh.surfVerts.size();
    ipc.surface_Num    = tetMesh.surface.size();
    ipc.edge_Num       = tetMesh.surfEdges.size();
    ipc.tri_edge_num   = tetMesh.tri_edges.size();

    //ipc.IPC_dt = 0.01 / 1.0;//1.0 / 30;//1.0 / 100;
    ipc.MAX_CCD_COLLITION_PAIRS_NUM =
        1 * collision_detection_buff_scale
        * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
           * std::max((ipc.IPC_dt / 0.01), 2.0));
    ipc.MAX_COLLITION_PAIRS_NUM = (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                                  * 3 * collision_detection_buff_scale;

    ipc.triangleNum        = tetMesh.triangleNum;
    ipc.targetVert         = d_tetMesh.targetVert;
    ipc.targetInd          = d_tetMesh.targetIndex;
    ipc.softNum            = tetMesh.softNum;
    ipc.abd_fem_count_info = tetMesh.abd_fem_count_info;
    ipc.benchmark_compact_triplets =
        selected_scene == 7 && benchmark_cloth_case == "stack10"
        && ipc.surf_vertexNum > 200000;
    if(ipc.benchmark_compact_triplets)
    {
        const int original_ccd_capacity = ipc.MAX_CCD_COLLITION_PAIRS_NUM;
        ipc.MAX_CCD_COLLITION_PAIRS_NUM =
            std::min(ipc.MAX_CCD_COLLITION_PAIRS_NUM,
                     ipc.MAX_COLLITION_PAIRS_NUM);
        std::cout << "BENCHMARK_CCD_CAP original=" << original_ccd_capacity
                  << " reserved=" << ipc.MAX_CCD_COLLITION_PAIRS_NUM
                  << std::endl;
    }

    std::cout << "ABD FEM count info: \n"
              << ipc.abd_fem_count_info << std::endl;


    printf("vertNum: %d      tetraNum: %d      faceNum: %d\n",
           ipc.vertexNum,
           ipc.tetrahedraNum,
           ipc.surface_Num);
    printf("surfVertNum: %d      surfEdgesNum: %d\n", ipc.surf_vertexNum, ipc.edge_Num);
    printf("maxCollisionPairsNum_CCD: %d      maxCollisionPairsNum: %d\n",
           ipc.MAX_CCD_COLLITION_PAIRS_NUM,
           ipc.MAX_COLLITION_PAIRS_NUM);

    //ipc.USE_MAS = false;
    ipc.MALLOC_DEVICE_MEM();

    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._faces, tetMesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._edges, tetMesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts,
                              tetMesh.surfVerts.data(),
                              ipc.surf_vertexNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    // Keep FEM body IDs used by the solver at -1. Only the BVH sees a
    // separate table group, excluding table-table VF/EE and CCD candidates.
    if(selected_scene == 7 && benchmark_cloth_case == "table"
       && !benchmark_object_manifest_path.empty()
       && benchmark_object_manifest.at("objects").at(0).at("self_collision") == false)
    {
        std::vector<int> collision_ids = tetMesh.point_id_to_body_id;
        const auto& table = cipc_components.at(0);
        const int begin = table.at("begin").get<int>();
        const int end = table.at("end").get<int>();
        if(begin < 0 || end > static_cast<int>(collision_ids.size()) || begin >= end)
            throw std::runtime_error("Invalid fixed table collision span");
        std::fill(collision_ids.begin() + begin, collision_ids.begin() + end, -2);
        CUDA_SAFE_CALL(cudaMalloc((void**)&benchmark_collision_body_ids,
                                  collision_ids.size() * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(benchmark_collision_body_ids, collision_ids.data(),
                                  collision_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    ipc.initBVH(d_tetMesh.BoundaryType,
                benchmark_collision_body_ids ? benchmark_collision_body_ids
                                             : d_tetMesh.point_id_to_body_id);

    if(ipc.pcg_data.P_type && true)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(ipc.vertexNum - tetMesh.abd_vertexOffset,
                                                    tetMesh.abd_vertexOffset,
                                                    neighborListSize,
                                                    ipc._collisonPairs,
                                                    tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                  tetMesh.neighborList.data(),
                                  neighborListSize * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                  tetMesh.neighborStart.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                  tetMesh.neighborNum.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                  tetMesh.partId_map_real.data(),
                                  tetMesh.part_offset * BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                  tetMesh.real_map_partId.data(),
                                  tetMesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));


#ifdef USE_QUADRATIC_BENDING
    // Precompute Q matrices for quadratic bending
    if(tetMesh.tri_edges.size() > 0)
    {
        printf("Precomputing Q matrices for quadratic bending (%zu edges)...\n",
               tetMesh.tri_edges.size());

        // Allocate host memory for Q matrices
        std::vector<Eigen::Matrix4d> Q_host(tetMesh.tri_edges.size());

        // Download rest positions, edges, and adjacency from device
        std::vector<double3> rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  d_tetMesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));

        // Call precomputation function (defined in femEnergy.cu)
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tetMesh.tri_edges.data(),
                            tetMesh.tri_edges_adj_points.data(),  // CPU端叫tri_edges_adj_points
                            tetMesh.tri_edges.size(),
                            Q_host.data());

        // Upload Q matrices to device
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.quad_bending_Q,
                                  Q_host.data(),
                                  tetMesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));

        printf("Quadratic bending Q matrices uploaded successfully.\n");

        // Optional: Print first Q matrix for verification
        if(tetMesh.tri_edges.size() > 0)
        {
            printf("First Q matrix:\n");
            for(int i = 0; i < 4; i++)
            {
                printf("  [%10.6f %10.6f %10.6f %10.6f]\n",
                       Q_host[0](i, 0),
                       Q_host[0](i, 1),
                       Q_host[0](i, 2),
                       Q_host[0](i, 3));
            }

            // Check for NaN or Inf
            int nan_count = 0;
            int inf_count = 0;
            for(size_t i = 0; i < Q_host.size(); i++)
            {
                for(int r = 0; r < 4; r++)
                {
                    for(int c = 0; c < 4; c++)
                    {
                        if(std::isnan(Q_host[i](r, c)))
                            nan_count++;
                        if(std::isinf(Q_host[i](r, c)))
                            inf_count++;
                    }
                }
            }
            if(nan_count > 0 || inf_count > 0)
            {
                printf("WARNING: Q matrices contain %d NaN values and %d Inf values!\n",
                       nan_count,
                       inf_count);
            }
        }
    }
#endif


    ipc.buildBVH();
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer, linear_system_buff_scale);

    printf("bboxDiagSize2: %f\n", ipc.bboxDiagSize2);
    printf("maxConer: %f  %f   %f           minCorner: %f  %f   %f\n",
           tetMesh.maxConer.x,
           tetMesh.maxConer.y,
           tetMesh.maxConer.z,
           tetMesh.minConer.x,
           tetMesh.minConer.y,
           tetMesh.minConer.z);

    printf("restSNKE: %f\n", ipc.RestNHEnergy);
    ipc.buildCP();

    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0 / motion_rate;
    //ipc.animation_fullRate = ipc.animation_subRate;
    ipc.computeXTilta(d_tetMesh, 1);
    ///////////////////////////////////////////////////////////////////////////////////

    ipc.create_LinearSystem(d_tetMesh);

    bvs.resize(2 * ipc.edge_Num - 1);
    nodes.resize(2 * ipc.edge_Num - 1);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(
        &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
}


void outputAnimationMeshInfo(std::string pathCloth, std::string pathBody)
{
    std::stringstream ss;
    ss << pathCloth;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    std::ofstream    outSurf(file_path);

    std::map<int, int> meshToSurf;
    for(int i = 0; i < bodyVertOffset; i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.triangles.size(); i++)
    {
        const auto& tri = tetMesh.triangles[i];
        outSurf << "f " << tri.x + 1 << " " << tri.y + 1 << " " << tri.z + 1 << std::endl;
    }
    outSurf.close();

    std::stringstream ss2;
    ss2 << pathBody;
    ss2.fill('0');
    ss2.width(5);
    ss2 << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss2 << ".obj";
    std::string file_path2 = ss2.str();
    std::ofstream    outSurf2(file_path2);

    //std::map<int, int> meshToSurf;
    for(int i = bodyVertOffset; i < tetMesh.vertexes.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf2 << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < clothFaceOffset; i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf2 << "f " << tri.x + 1 - bodyVertOffset << " " << tri.y + 1 - bodyVertOffset
                 << " " << tri.z + 1 - bodyVertOffset << std::endl;
    }
    outSurf2.close();
    surfNumId++;
}
bool pri = true;
void display(void)
{
    draw_Scene3D();
    std::filesystem::exists(std::string{gipc::output_dir()})
        || std::filesystem::create_directory(std::string{gipc::output_dir()});
    auto output_path = std::string{gipc::output_dir()} + "saveSurface/";

    std::filesystem::exists(output_path) || std::filesystem::create_directory(output_path);

    //if(stop)
    //    return;


    ipc.IPC_Solver(d_tetMesh);

    if(ipc.animation && true)
    {
        std::string filename =
            "triMesh/body4/postcvpr_big_body_" + std::to_string(frameId + 1) + ".obj";
        frameId++;
        tetMesh.load_animation(filename, 1, make_double3(-1, -0.5, -0.5));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                                  tetMesh.targetPos.data(),
                                  tetMesh.softNum * sizeof(double3),
                                  cudaMemcpyHostToDevice));
    }


    CUDA_SAFE_CALL(cudaMemcpy(
        &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                              ipc._vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToHost));


    if(screenshot)
    {
        std::stringstream ss;
        ss << "saveScreen/step_";
        ss.fill('0');
        ss.width(5);
        ss << step / 1;
        std::string file_path = ss.str();
        SaveScreenShot(window_width, window_height, file_path);
    }
    step++;
    printf("step:  %d\n", step);

    //if(step >= 160)
    //{
    //    std::cout << "step: " << step << " finished." << std::endl;
    //    exit(0);
    //}
}

void init(void)
{
    Init_CUDA();

    //main2();

    GLenum err = glewInit();
    if(GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    glClearColor(0.0, 0.0, 0.0, 1.0);


    LoadSettings();

    ipc.build_gipc_system(d_tetMesh);

    initScene();

    if(!isSetShader)
    {
        glViewport(0, 0, window_width, window_height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (float)window_width / window_height, 10.1f, 500.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    else
    {
        glGenBuffers(1, &PN_vbo_);
        glGenVertexArrays(1, &VAO);
    }
    //glEnable(GL_DEPTH_TEST);
}


void idle_func()
{
    glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
    //window_width = width;
    //window_height = height;

    glViewport(0, 0, width, height);
    if(!isSetShader)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        gluPerspective(45.0, (float)width / height, 0.1, 500.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    //glTranslatef(0.5f, 0.5f, -4.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
    if(key == 'w')
    {
        zTrans += .3f;
    }

    if(key == 's')
    {
        zTrans -= .3f;
    }

    if(key == 'a')
    {
        xTrans += .3f;
    }

    if(key == 'd')
    {
        xTrans -= .3f;
    }

    if(key == 'q')
    {
        yTrans -= .3f;
    }

    if(key == 'e')
    {
        yTrans += .3f;
    }

    if(key == '/')
    {
        screenshot = !screenshot;
    }

    if(key == '9')
    {
        saveSurface = !saveSurface;
    }

    if(key == 'k')
    {
        drawSurface = !drawSurface;
    }

    if(key == 'f')
    {
        drawbvh = !drawbvh;
    }

    if(key == ' ')
    {
        stop = !stop;
    }
    glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
    glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
    if(state == GLUT_DOWN)
    {
        buttonState = 1;
    }
    else if(state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if(buttonState == 1)
    {
        xRot += dy / 5.0f;
        yRot += dx / 5.0f;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}


void SpecialKey(GLint key, GLint x, GLint y)
{
    if(key == GLUT_KEY_DOWN)
    {
        change = true;
        initPath -= 1;
        if(initPath < 0)
        {
            initPath = obj_pathes.size() - 1;
        }
    }

    if(key == GLUT_KEY_UP)
    {
        change = true;
        initPath += 1;
        if(initPath == obj_pathes.size())
        {
            initPath = 0;
        }
    }
    glutPostRedisplay();
}


int main(int argc, char** argv) try
{
    int benchmark_frames = 0;
    bool benchmark_initialize_only = false;
    std::string requested_benchmark_output;
    bool benchmark_export_states = false;
    bool benchmark_export_physics = false;
    bool benchmark_export_object_audit = false;
    for(int i = 1; i < argc; ++i)
    {
        if(std::string{argv[i]} == "--scene" && i + 1 < argc)
            selected_scene = std::atoi(argv[++i]);
        else if(std::string{argv[i]} == "--benchmark-frames" && i + 1 < argc)
            benchmark_frames = std::atoi(argv[++i]);
        else if(std::string{argv[i]} == "--benchmark-output-dir" && i + 1 < argc)
            requested_benchmark_output = argv[++i];
        else if(std::string{argv[i]} == "--benchmark-initialize-only")
            benchmark_initialize_only = true;
        else if(std::string{argv[i]} == "--cloth-case" && i + 1 < argc)
            benchmark_cloth_case = argv[++i];
        else if(std::string{argv[i]} == "--benchmark-export-states")
            benchmark_export_states = true;
        else if(std::string{argv[i]} == "--benchmark-export-physics")
            benchmark_export_physics = true;
        else if(std::string{argv[i]} == "--benchmark-export-path-frame" && i + 1 < argc)
            benchmark_export_path_frame = std::stoi(argv[++i]);
        else if(std::string{argv[i]} == "--benchmark-export-object-audit")
            benchmark_export_object_audit = true;
        else if(std::string{argv[i]} == "--experimental-b-mode" && i + 1 < argc)
        {
            const std::string mode = argv[++i];
            if(mode == "defect_only") experimental_b_mode = 1;
            else if(mode == "adaptive_direct_residual") experimental_b_mode = 2;
            else if(mode == "full_b_nograph") experimental_b_mode = 3;
            else if(mode == "full_b_graph") experimental_b_mode = 4;
            else if(mode == "adaptive_direct_residual_graph") experimental_b_mode = 5;
            else if(mode == "full_b_dual_graph") experimental_b_mode = 6;
            else if(mode == "full_b_guarded_dual_graph") experimental_b_mode = 7;
            else if(mode == "full_b_budget_only_guarded") experimental_b_mode = 8;
            else if(mode == "full_b_conditional_mas_budget_guarded") experimental_b_mode = 9;
            else throw std::runtime_error("Unknown experimental B mode");
        }
        else if(std::string{argv[i]} == "--experimental-eps-r" && i + 1 < argc)
            experimental_eps_r = std::stod(argv[++i]);
        else if(std::string{argv[i]} == "--experimental-b-guard-multiplier" && i + 1 < argc)
            experimental_b_guard_multiplier = std::stod(argv[++i]);
        else if(std::string{argv[i]} == "--experimental-b-gradient-audit")
            experimental_b_gradient_audit = true;
        else if(std::string{argv[i]} == "--experimental-b-gradient-split")
            experimental_b_gradient_split = true;
        else if(std::string{argv[i]} == "--experimental-b-compatibility")
            experimental_b_strict = false;
        else if(std::string{argv[i]} == "--experimental-b-quiet-trace")
            experimental_b_quiet_trace = true;
        else if(std::string{argv[i]} == "--experimental-b-terminal-fast-stop")
            experimental_b_terminal_fast_stop = true;
        else if(std::string{argv[i]} == "--experimental-b-fused-defect-reduction")
            experimental_b_fused_defect_reduction = true;
        else if(std::string{argv[i]} == "--experimental-verified-fixed-pcg")
            experimental_verified_fixed_pcg = true;
        else if(std::string{argv[i]} == "--experimental-pcg-legacy-stop")
            experimental_pcg_legacy_stop = true;
        else if(std::string{argv[i]} == "--experimental-adaptive-pcg")
            experimental_adaptive_pcg = true;
        else if(std::string{argv[i]} == "--experimental-defect-export-vectors")
            experimental_defect_export_vectors = true;
        else if(std::string{argv[i]} == "--experimental-defect-shadow")
            experimental_defect_shadow = true;
        else if(std::string{argv[i]} == "--experimental-pcg-graph-cache")
            experimental_pcg_graph_cache = true;
        else if(std::string{argv[i]} == "--experimental-pcg-graph-update")
        {
            experimental_pcg_graph_update = true;
            experimental_pcg_graph_cache = true;
        }
        else if(std::string{argv[i]} == "--experimental-pcg-graph-tail-cache")
        {
            experimental_pcg_graph_tail_cache = true;
            experimental_pcg_graph_update = true;
            experimental_pcg_graph_cache = true;
        }
        else if(std::string{argv[i]} == "--experimental-pcg-graph-segment")
            experimental_pcg_graph_segment = true;
        else if(std::string{argv[i]} == "--experimental-device-scalar-pcg")
            experimental_device_scalar_pcg = true;
        else if(std::string{argv[i]} == "--experimental-pcg-conditional-while")
            experimental_pcg_conditional_while = true;
        else if(std::string{argv[i]} == "--experimental-pcg-conditional-mas")
        {
            experimental_pcg_conditional_while = true;
            experimental_pcg_conditional_mas = true;
        }
        else if(std::string{argv[i]} == "--experimental-pcg-conditional-cache")
            experimental_pcg_conditional_cache = true;
        else if(std::string{argv[i]} == "--experimental-pcg-fused-dot-tail")
            experimental_pcg_fused_dot_tail = true;
        else if(std::string{argv[i]} == "--experimental-pcg-fused-continue")
            experimental_pcg_fused_continue = true;
        else if(std::string{argv[i]} == "--experimental-intersection-scratch")
            experimental_intersection_scratch = true;
        else if(std::string{argv[i]} == "--experimental-mas-static-topology")
            experimental_mas_static_topology = true;
        else if(std::string{argv[i]} == "--experimental-mas-contact-topology")
            experimental_mas_contact_topology = true;
        else if(std::string{argv[i]} == "--experimental-mas-fused-clear")
            experimental_mas_fused_clear = true;
        else if(std::string{argv[i]} == "--experimental-reuse-newton-events")
            experimental_reuse_newton_events = true;
        else if(std::string{argv[i]} == "--experimental-batched-energy")
            experimental_batched_energy = true;
        else if(std::string{argv[i]} == "--experimental-energy-reuse")
            experimental_energy_reuse = true;
        else if(std::string{argv[i]} == "--experimental-energy-reuse-audit")
            experimental_energy_reuse_audit = true;
        else if(std::string{argv[i]} == "--experimental-ccd-bvh-refit")
            experimental_ccd_bvh_refit = true;
        else if(std::string{argv[i]} == "--experimental-ccd-bvh-refit-audit")
            experimental_ccd_bvh_refit_audit = true;
        else if(std::string{argv[i]} == "--experimental-graph-spmv")
            experimental_graph_spmv = true;
        else if(std::string{argv[i]} == "--scene-manifest" && i + 1 < argc)
            benchmark_scene_manifest_path = argv[++i];
        else if(std::string{argv[i]} == "--object-manifest" && i + 1 < argc)
            benchmark_object_manifest_path = argv[++i];
    }
    if(benchmark_initialize_only
       && (requested_benchmark_output.empty() || benchmark_export_states
           || benchmark_export_physics || benchmark_export_object_audit))
        throw std::runtime_error("Initialization-only requires an output directory and no frame exports");
    if(benchmark_export_path_frame != -1
       && (benchmark_export_path_frame < 1
           || benchmark_export_path_frame > benchmark_frames
           || requested_benchmark_output.empty()))
        throw std::runtime_error("Path export requires a valid benchmark frame and output");
    if(experimental_b_gradient_audit && !experimental_b_gradient_split)
        throw std::runtime_error("gradient audit requires gradient split");
    if(experimental_b_gradient_split && !experimental_b_mode)
        throw std::runtime_error("gradient split requires B mode");
    if(experimental_b_quiet_trace && !experimental_b_mode)
        throw std::runtime_error("quiet B trace requires B mode");
    if(experimental_b_fused_defect_reduction
       && (experimental_b_mode != 8 && experimental_b_mode != 9))
        throw std::runtime_error("fused defect reduction requires guarded B");
    if(experimental_b_terminal_fast_stop
       && (experimental_b_mode != 8 && experimental_b_mode != 9
           || experimental_b_gradient_split || experimental_b_gradient_audit
           || experimental_defect_export_vectors))
        throw std::runtime_error("B terminal fast stop requires guarded B without gradient audit/export");
    if(experimental_b_mode)
    {
        if(!std::isfinite(experimental_eps_r) || experimental_eps_r <= 0)
            throw std::runtime_error("B mode requires finite positive --experimental-eps-r");
        if((experimental_b_mode == 8 || experimental_b_mode == 9)
           && (!std::isfinite(experimental_b_guard_multiplier)
               || experimental_b_guard_multiplier < 1.0))
            throw std::runtime_error("Invalid guarded B diagnostic factors");
        if(experimental_adaptive_pcg || experimental_verified_fixed_pcg
           || experimental_pcg_graph_cache || experimental_pcg_graph_segment
           || experimental_device_scalar_pcg || experimental_graph_spmv
           || experimental_pcg_conditional_while)
            throw std::runtime_error("B mode controls its own PCG and graph flags");
        if(experimental_b_mode == 1) experimental_verified_fixed_pcg = true;
        else if(experimental_b_mode == 8 || experimental_b_mode == 9)
            experimental_defect_shadow = true;
        else experimental_adaptive_pcg = true;
        if(experimental_b_mode == 9)
        {
            experimental_pcg_conditional_while = true;
            experimental_pcg_conditional_mas = true;
        }
        if(experimental_b_mode == 4 || experimental_b_mode == 5
           || experimental_b_mode == 6 || experimental_b_mode == 7
           || experimental_b_mode == 8)
            experimental_pcg_graph_cache = true;
        if(experimental_b_mode == 6 || experimental_b_mode == 7
           || experimental_b_mode == 8)
            experimental_pcg_graph_tail_cache = true;
    }
    if(experimental_pcg_conditional_cache
       && !experimental_pcg_conditional_mas)
        throw std::runtime_error("Conditional graph cache requires MAS graph mode");
    if(experimental_pcg_fused_dot_tail
       && !experimental_pcg_conditional_cache)
        throw std::runtime_error("Fused PCG dot tail requires cached conditional graph");
    if(experimental_pcg_fused_continue
       && !experimental_pcg_conditional_cache)
        throw std::runtime_error("Fused PCG continuation requires cached conditional graph");
    if(experimental_intersection_scratch
       && !experimental_pcg_conditional_cache)
        throw std::runtime_error("Intersection scratch requires cached MAS graph mode");
    if(experimental_mas_static_topology
       && !experimental_pcg_conditional_cache)
        throw std::runtime_error("Static MAS topology requires cached MAS graph mode");
    if(experimental_mas_contact_topology && !experimental_mas_static_topology)
        throw std::runtime_error("Contact MAS topology requires static topology mode");
    if(experimental_mas_fused_clear && !experimental_pcg_conditional_mas)
        throw std::runtime_error("Fused MAS clear requires conditional MAS mode");
    if(experimental_energy_reuse && (!experimental_pcg_conditional_mas
                                     || !experimental_batched_energy))
        throw std::runtime_error("Energy reuse requires conditional MAS and batched energy");
    if(experimental_energy_reuse_audit && !experimental_energy_reuse)
        throw std::runtime_error("Energy reuse audit requires energy reuse");
    if(experimental_ccd_bvh_refit && !experimental_pcg_conditional_mas)
        throw std::runtime_error("CCD BVH refit requires conditional MAS mode");
    if(experimental_ccd_bvh_refit_audit && !experimental_ccd_bvh_refit)
        throw std::runtime_error("CCD BVH refit audit requires refit");
    if(experimental_adaptive_pcg || experimental_verified_fixed_pcg)
        experimental_defect_shadow = true;
    if(experimental_adaptive_pcg && experimental_verified_fixed_pcg)
        throw std::runtime_error("Choose adaptive or verified-fixed PCG");
    if(experimental_pcg_legacy_stop && !experimental_verified_fixed_pcg)
        throw std::runtime_error("Legacy PCG stop prototype requires verified-fixed PCG");
    if((experimental_adaptive_pcg || experimental_verified_fixed_pcg)
       && (experimental_graph_spmv || experimental_device_scalar_pcg
           || experimental_pcg_graph_segment
           || (experimental_pcg_graph_cache && experimental_b_mode != 4
               && experimental_b_mode != 5 && experimental_b_mode != 6
               && experimental_b_mode != 7 && experimental_b_mode != 8)))
        throw std::runtime_error("Adaptive PCG prototype must run without graph modes");
    if(experimental_pcg_conditional_while
       && (experimental_graph_spmv || experimental_device_scalar_pcg
           || experimental_pcg_graph_segment || experimental_pcg_graph_cache
           || experimental_adaptive_pcg
           || (experimental_verified_fixed_pcg
               && !(experimental_pcg_legacy_stop
                    && experimental_pcg_conditional_mas))
           || (experimental_defect_shadow && experimental_b_mode != 9
               && !(experimental_verified_fixed_pcg
                    && experimental_pcg_legacy_stop
                    && experimental_pcg_conditional_mas))))
        throw std::runtime_error("Conditional PCG graph requires isolated fixed PCG mode");
    if((experimental_graph_spmv && experimental_device_scalar_pcg)
       || (experimental_pcg_graph_segment &&
           (experimental_graph_spmv || experimental_device_scalar_pcg))
       || (experimental_pcg_graph_cache &&
           (experimental_graph_spmv || experimental_device_scalar_pcg
            || experimental_pcg_graph_segment)))
        throw std::runtime_error("Experimental PCG/SpMV modes must run separately");
    if(experimental_defect_export_vectors && !experimental_defect_shadow)
        throw std::runtime_error("defect vector export requires defect shadow");
    if(!benchmark_scene_manifest_path.empty())
    {
        if(requested_benchmark_output.empty())
            throw std::runtime_error("--scene-manifest requires --benchmark-output-dir");
        std::ifstream source(benchmark_scene_manifest_path);
        if(!source || !(source >> benchmark_scene_manifest))
            throw std::runtime_error("Cannot parse benchmark scene manifest");
    const std::set<std::string> known_scene_keys{"canonical_effective_scene_sha256", "canonical_source", "case_id", "case_spec", "complete_scene_manifest", "effective_scalar_fields", "initial_arrays", "initial_counts", "schema_version", "unresolved_fields"};
    const std::set<std::string> known_scalar_keys{"active_implicit_planes", "bbox_diagonal_squared", "bending_young", "cloth_bending_model", "cloth_bending_stiffness", "cloth_density", "cloth_shear_stiffness", "cloth_stretch_model", "cloth_stretch_stiffness", "cloth_thickness", "cloth_young", "density", "dhat_absolute", "dhat_squared", "dt", "friction_compiled", "friction_rate", "ground_friction_rate", "implicit_plane_rule", "newton_direction_threshold_coefficient", "pcg_requested_threshold", "poisson_ratio", "preconditioner_type", "relative_dhat", "strain_rate"};
    if(benchmark_scene_manifest.size() != known_scene_keys.size())
        throw std::runtime_error("Unknown or missing scene manifest top-level key");
    for(auto item = benchmark_scene_manifest.begin(); item != benchmark_scene_manifest.end(); ++item)
        if(!known_scene_keys.count(item.key()))
            throw std::runtime_error("Unknown scene manifest top-level key: " + item.key());
    const auto& scene_fields = benchmark_scene_manifest.at("effective_scalar_fields");
    if(!scene_fields.is_object() || scene_fields.size() != known_scalar_keys.size())
        throw std::runtime_error("Unknown or missing scene manifest scalar");
    for(auto item = scene_fields.begin(); item != scene_fields.end(); ++item)
        if(!known_scalar_keys.count(item.key()))
            throw std::runtime_error("Unknown scene manifest scalar: " + item.key());
        if(benchmark_scene_manifest.at("schema_version") !=
               "stiff4-canonical-scene-preflight-v1"
           || benchmark_scene_manifest.at("complete_scene_manifest") != false
           || benchmark_scene_manifest.at("case_spec").at("scene_id").get<int>() != selected_scene
           || benchmark_scene_manifest.at("case_spec").at("cloth_case") !=
                  (selected_scene == 7 ? gipc::Json(benchmark_cloth_case) : gipc::Json(nullptr)))
            throw std::runtime_error("Benchmark scene manifest identity mismatch");
    }
    if(!benchmark_object_manifest_path.empty())
    {
        if(benchmark_scene_manifest_path.empty())
            throw std::runtime_error("Object construction manifest requires a shared scene");
        std::ifstream source(benchmark_object_manifest_path);
        if(!source || !(source >> benchmark_object_manifest))
            throw std::runtime_error("Cannot parse object construction manifest");
        const std::set<std::string> keys{
            "case_id", "complete_scene_manifest", "objects", "schema_version"};
        const bool v2 = benchmark_object_manifest.at("schema_version") ==
                        "stiff4-object-construction-v2";
        const std::set<std::string> v1_keys{
            "body_type", "dimension", "name", "robust_mesh", "source_scale",
            "stiff_mesh", "stiff_mesh_sha256", "robust_mesh_sha256",
            "tet_young_pa", "world_translation_m"};
        const std::set<std::string> v2_keys{
            "body_type", "dimension", "name", "robust_mesh", "stiff_mesh",
            "stiff_mesh_sha256", "robust_mesh_sha256", "source_transform_f64x16",
            "instance_transforms_f64x16", "young_pa", "material_model",
            "fixed_mode", "self_collision"};
        if(benchmark_object_manifest.size() != keys.size())
            throw std::runtime_error("Unknown or missing object manifest field");
        for(auto item = benchmark_object_manifest.begin();
            item != benchmark_object_manifest.end(); ++item)
            if(!keys.count(item.key()))
                throw std::runtime_error("Unknown object manifest field: " + item.key());
        const int count = !v2 ? (selected_scene == 1 ? 3 : selected_scene == 6 ? 2 : 0)
            : selected_scene == 5 ? 3
            : selected_scene == 7 ? (benchmark_cloth_case == "stack10" ? 10
                                     : benchmark_cloth_case == "hang" ? 1 : 2) : 0;
        if(count == 0 || (!v2 && benchmark_object_manifest.at("schema_version") !=
                          "stiff4-object-construction-v1")
           || benchmark_object_manifest.at("case_id") !=
                  benchmark_scene_manifest.at("case_id")
           || benchmark_object_manifest.at("complete_scene_manifest") != false
           || !benchmark_object_manifest.at("objects").is_array()
           || benchmark_object_manifest.at("objects").size() != count)
            throw std::runtime_error("Object manifest identity mismatch");
        const auto check_matrix = [](const gipc::Json& values) {
            if(!values.is_array() || values.size() != 16)
                throw std::runtime_error("Invalid object matrix size");
            for(const auto& value : values)
                if(!std::isfinite(value.get<double>()))
                    throw std::runtime_error("Non-finite object matrix");
            for(int i = 0; i < 4; ++i)
                if(std::abs(values.at(12 + i).get<double>() - (i == 3 ? 1.0 : 0.0)) > 1e-12)
                    throw std::runtime_error("Non-affine object matrix");
        };
        for(const auto& object : benchmark_object_manifest.at("objects"))
        {
            const auto& allowed = v2 ? v2_keys : v1_keys;
            if(!object.is_object() || object.size() != allowed.size())
                throw std::runtime_error("Unknown or missing object field");
            for(auto item = object.begin(); item != object.end(); ++item)
                if(!allowed.count(item.key()))
                    throw std::runtime_error("Unknown object field: " + item.key());
            if(v2)
            {
                check_matrix(object.at("source_transform_f64x16"));
                const auto& instances = object.at("instance_transforms_f64x16");
                if(!instances.is_array() || instances.empty())
                    throw std::runtime_error("Missing object instances");
                for(const auto& instance : instances) check_matrix(instance);
                const auto material = object.at("material_model").get<std::string>();
                const auto fixed = object.at("fixed_mode").get<std::string>();
                if((material != "AffineBody" && material != "BaraffWitkinShell"
                    && material != "NeoHookeanShell")
                   || (fixed != "none" && fixed != "all"
                       && fixed != "diagonal_corners" && fixed != "edge_x")
                   || !object.at("self_collision").is_boolean())
                    throw std::runtime_error("Invalid object physics fields");
            }
            else
            {
                const auto& shift = object.at("world_translation_m");
                const double scale = object.at("source_scale").get<double>();
                if(!shift.is_array() || shift.size() != 3 || !std::isfinite(scale)
                   || scale <= 0)
                    throw std::runtime_error("Invalid object transform");
                for(const auto& coordinate : shift)
                    if(!std::isfinite(coordinate.get<double>()))
                        throw std::runtime_error("Non-finite object translation");
            }
            const auto mesh = std::filesystem::path(
                object.at("stiff_mesh").get<std::string>());
            if(mesh.is_absolute())
                throw std::runtime_error("Absolute Stiff object mesh path");
            const auto repo_root = std::filesystem::weakly_canonical(
                std::filesystem::path(assets_dir) / "..");
            const auto asset = std::filesystem::weakly_canonical(
                std::filesystem::path(assets_dir) / mesh);
            const auto inside = asset.lexically_relative(repo_root);
            if(inside.empty() || *inside.begin() == "..")
                throw std::runtime_error("Object mesh path escapes workspace");
        }
    }
    if((benchmark_export_states || benchmark_export_physics
       || benchmark_export_path_frame != -1
       || benchmark_export_object_audit || experimental_graph_spmv
       || experimental_device_scalar_pcg || experimental_pcg_graph_segment
       || experimental_pcg_graph_cache || experimental_defect_shadow
       || experimental_pcg_conditional_while)
       && requested_benchmark_output.empty())
    {
        std::cerr << "state export and graph prototype require --benchmark-output-dir\n";
        return 2;
    }
    if(!requested_benchmark_output.empty())
    {
        if((benchmark_initialize_only ? benchmark_frames != 0 : benchmark_frames <= 0)
           || selected_scene < 0 || selected_scene > 7
           || (benchmark_export_states && requested_benchmark_output.empty())
           || (selected_scene == 7 && benchmark_cloth_case.empty())
           || (selected_scene != 7 && !benchmark_cloth_case.empty()))
        {
            std::cerr << "benchmark output requires positive frames (or initialization-only with zero frames), scene 0..7, and --cloth-case for scene 7 only\n";
            return 2;
        }
        const auto out = std::filesystem::absolute(requested_benchmark_output);
        if(std::filesystem::exists(out / "frames.csv")
           || std::filesystem::exists(out / "status.json"))
        {
            std::cerr << "benchmark output already contains a run: " << out << '\n';
            return 2;
        }
        std::filesystem::create_directories(out);
        benchmark_output_dir = out.string();
        gipc::Json requested;
        requested["schema_version"] = "stiff4-m1-diagnostic-v1";
        requested["method_id"] =
            experimental_b_mode == 1 ? "defect_only_prototype" :
            experimental_b_mode == 2 ? "adaptive_direct_residual_prototype" :
            experimental_b_mode == 3 && experimental_b_gradient_split
                ? "full_b_gradient_split_prototype" :
            experimental_b_mode == 3 ? "full_b_nograph_prototype" :
            experimental_b_mode == 4 ? "full_b_graph_prototype" :
            experimental_b_mode == 5 ? "adaptive_direct_residual_graph_prototype" :
            experimental_b_mode == 6 ? "full_b_dual_graph_prototype" :
            experimental_b_mode == 7 ? "full_b_guarded_dual_graph_prototype" :
            experimental_b_mode == 8 ? "full_b_budget_only_guarded_prototype" :
            experimental_b_mode == 9 && experimental_batched_energy
                && experimental_mas_static_topology
                ? "full_b_conditional_mas_budget_cached_topology_energy_batch_prototype" :
            experimental_b_mode == 9 && experimental_reuse_newton_events
                && experimental_mas_static_topology
                ? "full_b_conditional_mas_budget_cached_topology_event_reuse_prototype" :
            experimental_b_mode == 9 && experimental_mas_contact_topology
                ? "full_b_conditional_mas_budget_cached_contact_topology_prototype" :
            experimental_b_mode == 9 && experimental_mas_static_topology
                ? "full_b_conditional_mas_budget_cached_topology_prototype" :
            experimental_b_mode == 9 && experimental_intersection_scratch
                ? "full_b_conditional_mas_budget_cached_scratch_prototype" :
            experimental_b_mode == 9 && experimental_pcg_conditional_cache
                ? "full_b_conditional_mas_budget_cached_prototype" :
            experimental_b_mode == 9 ? "full_b_conditional_mas_budget_guarded_prototype" :
            experimental_adaptive_pcg ? "adaptive_only_prototype" :
            experimental_verified_fixed_pcg && experimental_pcg_legacy_stop
                && experimental_pcg_conditional_mas
                ? "legacy_verified_conditional_mas_prototype" :
            experimental_verified_fixed_pcg && experimental_pcg_legacy_stop
                ? "legacy_verified_fixed_pcg_prototype" :
            experimental_verified_fixed_pcg ? "verified_fixed_pcg_prototype" :
            experimental_defect_shadow && experimental_pcg_graph_cache
                ? "pcg_graph_cache_defect_shadow_prototype" :
            experimental_defect_shadow ? "defect_shadow_prototype" :
            experimental_pcg_graph_tail_cache ? "pcg_graph_dual_segment_prototype" :
            experimental_pcg_graph_update ? "pcg_graph_update_prototype" :
            experimental_pcg_graph_cache ? "pcg_graph_cache_prototype" :
            experimental_pcg_graph_segment ? "pcg_graph_segment_prototype" :
            experimental_pcg_conditional_mas && experimental_batched_energy
                && experimental_mas_static_topology
                ? "pcg_conditional_mas_cached_topology_energy_batch_prototype" :
            experimental_pcg_conditional_mas && experimental_reuse_newton_events
                && experimental_mas_static_topology
                ? "pcg_conditional_mas_cached_topology_event_reuse_prototype" :
            experimental_pcg_conditional_mas && experimental_mas_contact_topology
                ? "pcg_conditional_mas_cached_contact_topology_prototype" :
            experimental_pcg_conditional_mas && experimental_mas_static_topology
                ? "pcg_conditional_mas_cached_topology_prototype" :
            experimental_pcg_conditional_mas && experimental_intersection_scratch
                ? "pcg_conditional_mas_cached_scratch_prototype" :
            experimental_pcg_conditional_mas && experimental_pcg_conditional_cache
                ? "pcg_conditional_mas_cached_prototype" :
            experimental_pcg_conditional_mas ? "pcg_conditional_mas_prototype" :
            experimental_pcg_conditional_while ? "pcg_conditional_while_prototype" :
            experimental_device_scalar_pcg ? "device_scalar_nograph" :
            experimental_graph_spmv ? "graph_spmv_prototype" :
            experimental_reuse_newton_events ? "base_newton_event_reuse_prototype" : "base";
        requested["scene_id"] = selected_scene;
        if(selected_scene == 7)
            requested["cloth_case"] = benchmark_cloth_case;
        requested["frames"] = benchmark_frames;
        requested["initialize_only"] = benchmark_initialize_only;
        requested["export_states"] = benchmark_export_states;
        requested["export_physics"] = benchmark_export_physics;
        requested["export_path_frame_one_based"] = benchmark_export_path_frame;
        requested["export_object_audit"] = benchmark_export_object_audit;
        requested["experimental_graph_spmv"] = experimental_graph_spmv;
        requested["experimental_device_scalar_pcg"] = experimental_device_scalar_pcg;
        requested["experimental_pcg_graph_segment"] = experimental_pcg_graph_segment;
        requested["experimental_pcg_graph_cache"] = experimental_pcg_graph_cache;
        requested["experimental_pcg_graph_update"] = experimental_pcg_graph_update;
        requested["experimental_pcg_graph_tail_cache"] = experimental_pcg_graph_tail_cache;
        requested["experimental_pcg_conditional_while"] = experimental_pcg_conditional_while;
        requested["experimental_pcg_conditional_mas"] = experimental_pcg_conditional_mas;
        requested["experimental_pcg_conditional_cache"] = experimental_pcg_conditional_cache;
        requested["experimental_pcg_fused_dot_tail"] = experimental_pcg_fused_dot_tail;
        requested["experimental_pcg_fused_continue"] = experimental_pcg_fused_continue;
        requested["experimental_intersection_scratch"] = experimental_intersection_scratch;
        requested["experimental_mas_static_topology"] = experimental_mas_static_topology;
        requested["experimental_mas_contact_topology"] = experimental_mas_contact_topology;
        requested["experimental_mas_fused_clear"] = experimental_mas_fused_clear;
        requested["experimental_reuse_newton_events"] = experimental_reuse_newton_events;
        requested["experimental_batched_energy"] = experimental_batched_energy;
        requested["experimental_energy_reuse"] = experimental_energy_reuse;
        requested["experimental_energy_reuse_audit"] = experimental_energy_reuse_audit;
        requested["experimental_ccd_bvh_refit"] = experimental_ccd_bvh_refit;
        requested["experimental_ccd_bvh_refit_audit"] = experimental_ccd_bvh_refit_audit;
        requested["experimental_defect_shadow"] = experimental_defect_shadow;
        requested["experimental_adaptive_pcg"] = experimental_adaptive_pcg;
        requested["experimental_verified_fixed_pcg"] = experimental_verified_fixed_pcg;
        requested["experimental_pcg_legacy_stop"] = experimental_pcg_legacy_stop;
        requested["experimental_pcg_true_residual_factor"] =
            experimental_pcg_legacy_stop ? 2.0 : 1.0;
        requested["experimental_b_mode"] = experimental_b_mode;
        requested["experimental_eps_r"] = experimental_b_mode ? gipc::Json(experimental_eps_r) : gipc::Json(nullptr);
        requested["experimental_b_guard_multiplier"] = (experimental_b_mode == 8 || experimental_b_mode == 9) ? gipc::Json(experimental_b_guard_multiplier) : gipc::Json(nullptr);
        requested["experimental_b_strict"] = experimental_b_strict;
        requested["experimental_b_gradient_split"] = experimental_b_gradient_split;
        requested["experimental_b_terminal_fast_stop"] = experimental_b_terminal_fast_stop;
        requested["experimental_b_fused_defect_reduction"] = experimental_b_fused_defect_reduction;
        requested["experimental_b_gradient_audit"] = experimental_b_gradient_audit;
        requested["experimental_b_quiet_trace"] = experimental_b_quiet_trace;
        requested["experimental_defect_export_vectors"] = experimental_defect_export_vectors;
        requested["object_manifest_path"] = benchmark_object_manifest_path.empty()
                                                ? gipc::Json(nullptr)
                                                : gipc::Json(std::filesystem::absolute(
                                                      benchmark_object_manifest_path).string());
        requested["scene_manifest_path"] = benchmark_scene_manifest_path.empty()
                                               ? gipc::Json(nullptr)
                                               : gipc::Json(std::filesystem::absolute(
                                                     benchmark_scene_manifest_path).string());
        requested["output_dir"] = benchmark_output_dir;
        std::ofstream(benchmark_output_dir + "/requested.json") << requested.dump(2);
    }
    if(benchmark_frames > 0 || benchmark_initialize_only)
    {
        std::filesystem::create_directories(std::string{gipc::output_dir()});
        benchmark_scene_id = selected_scene;
        Init_CUDA();
        LoadSettings();
        ipc.build_gipc_system(d_tetMesh);
        initScene(benchmark_initialize_only);
        if(benchmark_export_object_audit)
        {
            const auto& counts = tetMesh.abd_fem_count_info;
            auto& abd = ipc.m_abd_sim_data->device;
            const size_t body_count = counts.abd_body_num;
            if(abd.body_id_to_abd_mass.size() != body_count
               || abd.body_id_to_q.size() != body_count)
                throw std::runtime_error("ABD object audit count mismatch");
            std::vector<gipc::ABDJacobiDyadicMass> masses;
            std::vector<gipc::Vector12> q;
            abd.body_id_to_abd_mass.copy_to(masses);
            abd.body_id_to_q.copy_to(q);
            auto write_binary = [&](const std::string& file, const auto& values) {
                std::ofstream output(benchmark_output_dir + "/" + file,
                                     std::ios::binary | std::ios::trunc);
                output.write(reinterpret_cast<const char*>(values.data()),
                             static_cast<std::streamsize>(values.size()
                                                          * sizeof(values[0])));
                if(!output)
                    throw std::runtime_error("Cannot write object audit: " + file);
            };
            write_binary("object_point_body_ids.i32", tetMesh.point_id_to_body_id);
            write_binary("object_tet_body_ids.i32", tetMesh.tet_id_to_body_id);
            write_binary("object_tet_young.f64", tetMesh.vert_youngth_modules);
            std::vector<int> body_boundaries;
            body_boundaries.reserve(tetMesh.body_id_to_is_fixed.size());
            for(auto boundary : tetMesh.body_id_to_is_fixed)
                body_boundaries.push_back(static_cast<int>(boundary));
            write_binary("object_body_boundaries.i32", body_boundaries);
            std::vector<double> matrix_values;
            matrix_values.reserve(body_count * 144);
            gipc::Json bodies = gipc::Json::array();
            for(size_t index = 0; index < body_count; ++index)
            {
                const auto matrix = masses[index].to_mat();
                for(int row = 0; row < 12; ++row)
                    for(int col = 0; col < 12; ++col)
                        matrix_values.push_back(matrix(row, col));
                bodies.push_back({{"body_id", index},
                                  {"mass_kg", masses[index].mass()},
                                  {"center_world_m", {q[index](0), q[index](1), q[index](2)}},
                                  {"boundary_type", body_boundaries.at(index)}});
            }
            write_binary("object_abd_mass_matrices.f64x144", matrix_values);
            gipc::Json audit = {
                {"schema_version", "stiff4-m1-object-audit-v1"},
                {"stage", "initialized GPU state before first simulation frame"},
                {"abd_body_count", body_count},
                {"fem_body_count", counts.fem_body_num},
                {"abd_point_count", counts.abd_point_num},
                {"abd_tet_count", counts.abd_tet_num},
                {"mass_density_kg_m3", ipc.m_abd_system->parms.mass_density},
                {"generalized_mass_order", "q=(p_x,p_y,p_z,A_col0_xyz,A_col1_xyz,A_col2_xyz); 12x12 row-major"},
                {"point_body_ids", "object_point_body_ids.i32"},
                {"tet_body_ids", "object_tet_body_ids.i32"},
                {"tet_young", "object_tet_young.f64"},
                {"body_boundaries", "object_body_boundaries.i32"},
                {"abd_mass_matrices", "object_abd_mass_matrices.f64x144"},
                {"abd_bodies", bodies}};
            std::ofstream output(benchmark_output_dir + "/object_audit.json");
            output << audit.dump(2);
            if(!output)
                throw std::runtime_error("Cannot write object_audit.json");
        }
        if(!benchmark_output_dir.empty())
        {
            gipc::Json effective;
            effective["schema_version"] = "stiff4-m1-diagnostic-v1";
            effective["complete_scene_manifest"] = false;
            effective["experimental_graph_spmv"] = experimental_graph_spmv;
            effective["experimental_device_scalar_pcg"] = experimental_device_scalar_pcg;
            effective["experimental_pcg_graph_segment"] = experimental_pcg_graph_segment;
            effective["experimental_pcg_graph_cache"] = experimental_pcg_graph_cache;
            effective["experimental_pcg_graph_update"] = experimental_pcg_graph_update;
            effective["experimental_pcg_graph_tail_cache"] = experimental_pcg_graph_tail_cache;
            effective["experimental_pcg_conditional_while"] = experimental_pcg_conditional_while;
            effective["experimental_pcg_conditional_mas"] = experimental_pcg_conditional_mas;
            effective["experimental_pcg_conditional_cache"] = experimental_pcg_conditional_cache;
            effective["experimental_pcg_fused_dot_tail"] = experimental_pcg_fused_dot_tail;
            effective["experimental_pcg_fused_continue"] = experimental_pcg_fused_continue;
            effective["experimental_intersection_scratch"] = experimental_intersection_scratch;
            effective["experimental_mas_static_topology"] = experimental_mas_static_topology;
            effective["experimental_mas_contact_topology"] = experimental_mas_contact_topology;
            effective["experimental_mas_fused_clear"] = experimental_mas_fused_clear;
            effective["experimental_reuse_newton_events"] = experimental_reuse_newton_events;
            effective["experimental_batched_energy"] = experimental_batched_energy;
            effective["experimental_energy_reuse"] = experimental_energy_reuse;
            effective["experimental_energy_reuse_audit"] = experimental_energy_reuse_audit;
            effective["experimental_ccd_bvh_refit"] = experimental_ccd_bvh_refit;
            effective["experimental_ccd_bvh_refit_audit"] = experimental_ccd_bvh_refit_audit;
            effective["experimental_defect_shadow"] = experimental_defect_shadow;
            effective["experimental_adaptive_pcg"] = experimental_adaptive_pcg;
            effective["experimental_verified_fixed_pcg"] = experimental_verified_fixed_pcg;
            effective["experimental_pcg_legacy_stop"] = experimental_pcg_legacy_stop;
            effective["experimental_pcg_true_residual_factor"] =
                experimental_pcg_legacy_stop ? 2.0 : 1.0;
            effective["experimental_b_mode"] = experimental_b_mode;
            effective["experimental_eps_r"] = experimental_b_mode ? gipc::Json(experimental_eps_r) : gipc::Json(nullptr);
            effective["experimental_b_guard_multiplier"] = (experimental_b_mode == 8 || experimental_b_mode == 9) ? gipc::Json(experimental_b_guard_multiplier) : gipc::Json(nullptr);
            effective["experimental_b_strict"] = experimental_b_strict;
            effective["experimental_b_gradient_split"] = experimental_b_gradient_split;
            effective["experimental_b_terminal_fast_stop"] = experimental_b_terminal_fast_stop;
            effective["experimental_b_fused_defect_reduction"] = experimental_b_fused_defect_reduction;
            effective["experimental_b_gradient_audit"] = experimental_b_gradient_audit;
            effective["experimental_b_quiet_trace"] = experimental_b_quiet_trace;
            if(experimental_b_mode)
                effective["experimental_b_controller"] = {
                    {"min_accepted_updates", 6},
                    {"beta_tolerance", 1e-3},
                    {"relative_roundoff", 1e-11},
                    {"max_newton_iterations_diagnostic", 256},
                    {"near_convergence_pcg_factor", 10.0},
                    {"gradient_split_direction_gate_factor", 4.0},
                    {"eps_r_source", "explicit_pre_run_cli"}};
            if(experimental_adaptive_pcg)
                effective["adaptive_pcg_controller"] = {{"gamma_EW", 0.9}, {"p", 1.5},
                    {"eta_initial", 0.1}, {"eta_max", 0.1},
                    {"eta_min", std::sqrt(ipc.pcg_threshold)},
                    {"base_rho_rate", ipc.pcg_threshold},
                    {"true_residual_check", true},
                    {"baseline_resolve_on_failed_check", true},
                    {"baseline_confirm_on_legacy_stop", true}};
            effective["experimental_defect_export_vectors"] = experimental_defect_export_vectors;
            effective["fixed_table_self_collision_filtered"] =
                benchmark_collision_body_ids != nullptr
                || (benchmark_initialize_only && selected_scene == 7
                    && benchmark_cloth_case == "table"
                    && !benchmark_object_manifest_path.empty()
                    && benchmark_object_manifest.at("objects").at(0).at("self_collision") == false);
            effective["scene_manifest_consumed"] = !benchmark_scene_manifest_path.empty();
            effective["object_manifest_consumed"] = !benchmark_object_manifest_path.empty();
            effective["scene_id"] = selected_scene;
            if(selected_scene == 7)
                effective["cloth_case"] = benchmark_cloth_case;
            effective["frames"] = benchmark_frames;
            effective["dt"] = ipc.IPC_dt;
            effective["vertex_count"] = tetMesh.vertexNum;
            effective["tet_count"] = tetMesh.tetrahedraNum;
            effective["triangle_count"] = tetMesh.triangleNum;
            effective["abd_body_count"] = tetMesh.abd_fem_count_info.total_body_num();
            effective["fixed_or_constrained_vertex_count"] =
                std::count_if(tetMesh.boundaryTypies.begin(),
                              tetMesh.boundaryTypies.end(),
                              [](int type) { return type != 0; });
            effective["initial_mass_sum"] =
                std::accumulate(tetMesh.masses.begin(), tetMesh.masses.end(), 0.0);
            effective["density"] = ipc.density;
            effective["poisson_ratio"] = ipc.PoissonRate;
            effective["cloth_density"] = ipc.clothDensity;
            effective["cloth_thickness"] = ipc.clothThickness;
            effective["cloth_young"] = ipc.clothYoungModulus;
            effective["bending_young"] = ipc.bendYoungModulus;
            effective["strain_rate"] = ipc.strainRate;
            effective["cloth_stretch_stiffness"] = ipc.stretchStiff;
            effective["cloth_shear_stiffness"] = ipc.shearStiff;
            effective["cloth_bending_stiffness"] = ipc.bendStiff;
            effective["cloth_stretch_model"] = "BaraffWitkinStrainLimiting";
            effective["cloth_bending_model"] = "QuadraticBendingQ";
            effective["friction_rate"] = ipc.frictionRate;
            effective["ground_friction_rate"] = ipc.gd_frictionRate;
#ifdef USE_FRICTION
            effective["friction_compiled"] = true;
#else
            effective["friction_compiled"] = false;
#endif
            effective["newton_direction_threshold_coefficient"] =
                ipc.Newton_solver_threshold;
            effective["pcg_requested_threshold"] = ipc.pcg_threshold;
            effective["preconditioner_type"] = ipc.pcg_data.P_type;
            effective["relative_dhat"] = ipc.relative_dhat;
            effective["bbox_diagonal_squared"] = ipc.bboxDiagSize2;
            effective["dhat_squared"] = ipc.dHat;
            effective["dhat_absolute"] = std::sqrt(ipc.dHat);
            // GIPC allocates five plane slots, but all active collision kernels
            // dereference only slot zero. Report the actual effective boundary.
            effective["implicit_plane_slots_allocated"] = 5;
            effective["active_implicit_planes"] = gipc::Json::array(
                {{{"normal", {0.0, 1.0, 0.0}},
                  {"offset", selected_scene == 6 ? -1e6 : -1.0}}});
            effective["implicit_plane_rule"] = "dot(normal, x)-offset";
            auto write_initial_array = [&](const std::string& field,
                                           const std::string& file,
                                           const std::string& element_type,
                                           const auto& values) {
                std::ofstream output(benchmark_output_dir + "/" + file,
                                     std::ios::binary | std::ios::trunc);
                output.write(reinterpret_cast<const char*>(values.data()),
                             static_cast<std::streamsize>(values.size()
                                                          * sizeof(values[0])));
                if(!output)
                    throw std::runtime_error("Cannot write benchmark initial array: " + file);
                effective["initial_arrays"][field] = {
                    {"file", file},
                    {"count", values.size()},
                    {"element_type", element_type}};
            };
            write_initial_array("positions", "scene_positions.f64x3", "float64[3]",
                                tetMesh.vertexes);
            write_initial_array("velocities", "scene_velocities.f64x3", "float64[3]",
                                tetMesh.velocities);
            write_initial_array("triangles", "scene_triangles.u32x3", "uint32[3]",
                                tetMesh.triangles);
            write_initial_array("tetrahedra", "scene_tetrahedra.u32x4", "uint32[4]",
                                tetMesh.tetrahedras);
            write_initial_array("surface", "scene_surface.u32x3", "uint32[3]",
                                tetMesh.surface);
            write_initial_array("masses", "scene_masses.f64", "float64",
                                tetMesh.masses);
            write_initial_array("boundary_types", "scene_boundary_types.i32", "int32",
                                tetMesh.boundaryTypies);
            write_initial_array("apply_gravity", "scene_apply_gravity.i32", "int32",
                                tetMesh.apply_gravity);
            write_initial_array("target_indices", "scene_target_indices.u32", "uint32",
                                tetMesh.targetIndex);
            write_initial_array("target_positions", "scene_target_positions.f64x3", "float64[3]",
                                tetMesh.targetPos);
            effective["unresolved_scene_fields"] = {
                "canonical object/material IDs", "finite mesh contact boundary details",
                "per-object mass and inertia", "energy formula equivalence with robust"};
            std::ofstream effective_file(benchmark_output_dir + "/effective_scene.json");
            effective_file << effective.dump(2);
            if(!effective_file)
                throw std::runtime_error("Cannot write benchmark effective scene");
        }
        for(int frame = 0; frame < benchmark_frames; ++frame)
        {
            ipc.IPC_Solver(d_tetMesh);
            if(benchmark_export_states || benchmark_export_physics)
            {
                CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                          d_tetMesh.vertexes,
                                          tetMesh.vertexNum * sizeof(double3),
                                          cudaMemcpyDeviceToHost));
            }
            if(benchmark_export_states)
            {
                std::ofstream states(benchmark_output_dir + "/states.f64",
                                     std::ios::binary | std::ios::app);
                states.write(reinterpret_cast<const char*>(tetMesh.vertexes.data()),
                             tetMesh.vertexNum * sizeof(double3));
                if(!states)
                    throw std::runtime_error("Cannot write benchmark state");
            }
            if(benchmark_export_physics)
            {
                const double triangle_elastic =
                    ipc.Energy_Add_Reduction_Algorithm(8, d_tetMesh);
                const double bending_elastic =
                    ipc.Energy_Add_Reduction_Algorithm(10, d_tetMesh);
                CUDA_SAFE_CALL(cudaMemcpy(tetMesh.velocities.data(),
                                          d_tetMesh.velocities,
                                          tetMesh.vertexNum * sizeof(double3),
                                          cudaMemcpyDeviceToHost));
                std::ofstream velocities(benchmark_output_dir + "/velocities.f64",
                                         std::ios::binary | std::ios::app);
                velocities.write(reinterpret_cast<const char*>(tetMesh.velocities.data()),
                                 tetMesh.vertexNum * sizeof(double3));
                if(!velocities)
                    throw std::runtime_error("Cannot write benchmark velocities");
                double point_kinetic = 0.0;
                double gravity_potential = 0.0;
                for(size_t i = 0; i < tetMesh.vertexNum; ++i)
                {
                    const auto& v = tetMesh.velocities[i];
                    point_kinetic += 0.5 * tetMesh.masses[i]
                                     * (v.x * v.x + v.y * v.y + v.z * v.z);
                    if(tetMesh.apply_gravity[i])
                        gravity_potential +=
                            9.8 * tetMesh.masses[i] * tetMesh.vertexes[i].y;
                }
                if(!std::isfinite(triangle_elastic) || !std::isfinite(bending_elastic)
                   || !std::isfinite(point_kinetic)
                   || !std::isfinite(gravity_potential))
                    throw std::runtime_error("Non-finite benchmark physical component");
                std::ofstream physics(benchmark_output_dir + "/physics_components.csv",
                                      frame == 0 ? std::ios::trunc : std::ios::app);
                physics.precision(17);
                if(frame == 0)
                    physics << "frame,triangle_elastic_j,bending_elastic_j,point_kinetic_j,gravity_potential_j\n";
                physics << frame + 1 << ',' << triangle_elastic << ','
                        << bending_elastic << ',' << point_kinetic << ','
                        << gravity_potential << '\n';
                if(!physics)
                    throw std::runtime_error("Cannot write benchmark physics");
            }
        }
        if(!benchmark_output_dir.empty())
        {
            gipc::Json status;
            status["state"] = benchmark_initialize_only ? "initialized_diagnostic"
                : benchmark_newton_cap_hits == 0 ? "completed_diagnostic"
                                                : "newton_cap_hit";
            status["newton_cap_hits"] = benchmark_newton_cap_hits;
            status["completed_frames"] = benchmark_frames;
            status["timing_scope"] =
                "M1 instrumentation; logging and legacy in-loop synchronization remain";
            std::ofstream(benchmark_output_dir + "/status.json") << status.dump(2);
        }
        if(benchmark_collision_body_ids)
        {
            CUDA_SAFE_CALL(cudaFree(benchmark_collision_body_ids));
            benchmark_collision_body_ids = nullptr;
        }
        // A Newton cap is a failed B experiment even if output frames exist.
        const int benchmark_exit_code =
            experimental_b_mode && benchmark_newton_cap_hits ? 3 : 0;
#ifdef __linux__
        // CUDA global teardown can abort after all benchmark files are closed.
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(benchmark_exit_code);
#else
        return benchmark_exit_code;
#endif
    }
    glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

    glutSetOption(GLUT_MULTISAMPLE, 16);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("FEM");

    init();

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);


    glEnable(GL_MULTISAMPLE);
    glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);


    glutDisplayFunc(display);


    //glutDisplayFunc(display_func);
    glutReshapeFunc(reshape_func);
    glutKeyboardFunc(keyboard_func);
    glutSpecialFunc(&SpecialKey);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);


    glutMainLoop();
    //return 0;
}
catch(const std::exception& error)
{
    std::cerr << "STIFF4_SCENE_ERROR " << error.what() << '\n';
    return 2;
}



