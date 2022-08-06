#include <iostream>
#include <random>
#include "Frame.hpp"
#include "ceres_basolver.hpp"
#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <vertex_pose.hpp>
#include <vertex_landmark_pos.hpp>
#include <edge_reprojection_pos.hpp>

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(std::vector<Frame>& cameraPoses, std::vector<GraphOptimizor::Vector3<Scalar>>& points, int featureNums,int poseNums) 
{ 
    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 16); // 1/16 圆弧
        // 绕 z 轴 旋转
        GraphOptimizor::Matrix3<Scalar> R;
        R = Eigen::AngleAxis<Scalar>(theta, GraphOptimizor::Vector3<Scalar>::UnitZ());
        GraphOptimizor::Vector3<Scalar> t = GraphOptimizor::Vector3<Scalar>(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);

        GraphOptimizor::Vector3<Scalar> Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Pw);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            GraphOptimizor::Vector3<Scalar> Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            if (Pc.z() < 1e-10) {
                continue;
            }
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(std::make_pair(j, Pc));
        }
    }
}

void AddNoiseinCamera(std::vector<Frame> &cameras) 
{
    std::normal_distribution<double> camera_rotation_noise(0., 0.1);
    std::normal_distribution<double> camera_position_noise(0., 0.2);
    std::default_random_engine generator;

    for (size_t i = 0; i < cameras.size(); ++i) {
        // 给相机位置和姿态初值增加噪声
        GraphOptimizor::Matrix3<Scalar> noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitX());
        noise_Y = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitY());
        noise_Z = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        GraphOptimizor::Vector3<Scalar> noise_t(camera_position_noise(generator), camera_position_noise(generator), camera_position_noise(generator));
        GraphOptimizor::Quaternion<Scalar> noise_q_wc(noise_R);
        if (i < 2) {
            noise_t.setZero();
            noise_q_wc.setIdentity();
        }

        cameras[i].setNoise(noise_t, noise_q_wc);
        
    }
}

void GetSimNoisyLandmark(std::vector<GraphOptimizor::Vector3<Scalar>>& points,std::vector<GraphOptimizor::Vector3<Scalar>>& noised_pts)
{    for (size_t i = 0; i < points.size(); ++i) {
        // 为初值添加随机噪声
        std::default_random_engine generator;
        std::normal_distribution<double> landmark_position_noise(0., 0.5);
        GraphOptimizor::Vector3<Scalar> noise(landmark_position_noise(generator), landmark_position_noise(generator), landmark_position_noise(generator));
        //TODO check double/flaot
        GraphOptimizor::Vector3<Scalar> noised_pt = points[i] + noise;
        noised_pts.push_back(noised_pt);
    }
}


/* 程序主函数入口 */
int main() {
    std::cout << "Test GraphOptimizor solver on Mono BA problem." << std::endl;

    // 第一步：准备测试数据
    std::cout << "\nStep 1: Prepare dataset." << std::endl;
    std::vector<Frame> cameras, cameras_esti;
    std::vector<GraphOptimizor::Vector3<Scalar>> points, noisy_points, points_esti;
    GetSimDataInWordFrame(cameras, points,300,10);
    GetSimNoisyLandmark(points,noisy_points);
    AddNoiseinCamera(cameras);

    std::cout << "\nStep 1.5: Build Ceres problem." << std::endl;

    ceresBAProblem ceres_problem(cameras,noisy_points);
    ceres_problem.set_options();
    GraphOptimizor::Timer timer;
    ceres_problem.solve();
    std::cout << "<ceres> cost time " << timer.Stop() << std::endl;
    ceres_problem.generateResult(cameras_esti,points_esti);

    std::cout << "\nStep 1.8: Finish Ceres problem." << std::endl;


    // 第二步：构造待求解问题
    std::cout << "\nStep 2: Construct GraphOptimizor solver." << std::endl;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_cameraVertex = 0;
    size_t type_landmarkVertex = 1;

    // 第三步：构造相机 pose 节点，并添加到 problem 中
    std::cout << "\nStep 3: Add camera pose vertices." << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> camera_rotation_noise(0., 0.1);
    std::normal_distribution<double> camera_position_noise(0., 0.2);
    std::vector<std::shared_ptr<GraphOptimizor::VertexPose<Scalar>>> cameraVertices;
    for (size_t i = 0; i < cameras.size(); ++i) {
        // 给相机位置和姿态初值增加噪声
        GraphOptimizor::Matrix3<Scalar> noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitX());
        noise_Y = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitY());
        noise_Z = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        GraphOptimizor::Vector3<Scalar> noise_t(camera_position_noise(generator), camera_position_noise(generator), camera_position_noise(generator));
        GraphOptimizor::Quaternion<Scalar> noise_q_wc(noise_R);
        if (i < 2) {
            noise_t.setZero();
            noise_q_wc.setIdentity();
        }
        GraphOptimizor::Quaternion<Scalar> temp_q_wc = cameras[i].qwc * noise_q_wc;
        GraphOptimizor::Vector3<Scalar> temp_t_wc = cameras[i].twc + noise_t;

        GraphOptimizor::VectorX<Scalar> param(7);
        param << temp_t_wc, temp_q_wc.x(), temp_q_wc.y(), temp_q_wc.z(), temp_q_wc.w();
        std::shared_ptr<GraphOptimizor::VertexPose<Scalar>> cameraVertex(new GraphOptimizor::VertexPose<Scalar>());
        cameraVertex->SetParameters(param);
        cameraVertex->SetType(type_cameraVertex);
        cameraVertices.emplace_back(cameraVertex);
        problem.AddVertex(cameraVertex);
    }
    std::cout << "Add " << cameraVertices.size() << " camera vertices, problem has " << problem.GetVertexNum() << " vertices." << std::endl;

    // 第四步：构造特征点 position 节点，同时构造 reprojection 边，添加到 problem 中
    std::cout << "\nStep 4: Add landmark position vertices." << std::endl;
    std::vector<std::shared_ptr<GraphOptimizor::VertexLandmarkPosition<Scalar>>> landmarkVertices;
    std::normal_distribution<double> landmark_position_noise(0.0, 0.5);
    for (size_t i = 0; i < points.size(); ++i) {
        // 构造特征点的节点
        GraphOptimizor::VectorX<Scalar> param = points[i] + GraphOptimizor::Vector3<Scalar>(
            landmark_position_noise(generator), landmark_position_noise(generator), landmark_position_noise(generator));
        std::shared_ptr<GraphOptimizor::VertexLandmarkPosition<Scalar>> landmarkVertex(new GraphOptimizor::VertexLandmarkPosition<Scalar>());
        landmarkVertex->SetParameters(param);
        landmarkVertex->SetType(type_landmarkVertex);
        landmarkVertices.emplace_back(landmarkVertex);
        problem.AddVertex(landmarkVertex);

        // 遍历此点的观测，构造 reprojection 边，添加到 problem 中
        for (size_t j = 0; j < cameras.size(); ++j) {
            GraphOptimizor::Vector2<Scalar> normPoint = cameras[j].featurePerId.find(i)->second.head<2>();
            std::shared_ptr<GraphOptimizor::EdgeReprojectionPos<Scalar>> edge(new GraphOptimizor::EdgeReprojectionPos<Scalar>(normPoint));
            edge->AddVertex(landmarkVertex, 0);
            edge->AddVertex(cameraVertices[j], 1);
            std::shared_ptr<GraphOptimizor::TukeyKernel<Scalar>> kernel(new GraphOptimizor::TukeyKernel<Scalar>(1.0));
            edge->SetKernel(kernel);
            problem.AddEdge(edge);
        }
    }
    std::cout << "Add " << landmarkVertices.size() << " landmark vertices, problem has " << problem.GetVertexNum() << " vertices and " <<
        problem.GetEdgeNum() << " edges." << std::endl;

    // 第六步：配置相关参数，求解问题
    std::cout << "\nStep 6: Start solve problem." << std::endl;

    std::unique_ptr<tbb::global_control> ctrl =
        std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, 4);

    problem.SetMargnedVertexTypesWhenSolving(type_landmarkVertex);  // 设置在求解过程中需要暂时被边缘化的节点的类型
    problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::LM_Auto); // 设置数值优化方法
    // problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::DogLeg);
    problem.SetLinearSolver(GraphOptimizor::Problem<Scalar>::LinearSolver::PCG_Solver);
    // problem.LM_SetDampParameter(11.0, 10.0);    // 设置 LM 算法的阻尼因子调整参数
    GraphOptimizor::Problem<Scalar>::Options options;
    options.maxInvalidStep = 1;
    options.maxLambda = 1e32;
    options.maxMinCostHold = 1;
    options.maxRadius = 20;
    options.minCostDownRate = 1e-6;
    options.minLambda = 1e-8;
    options.minNormDeltaX = 1e-5;
    options.minPCGCostDownRate = 1e-6;
    options.minRadius = 1e-4;
    options.maxTimeCost = 20;
    problem.SetOptions(options);
    cameraVertices[0]->SetFixed();              // 因为是 VO 问题，固定前两帧相机 pose
    cameraVertices[1]->SetFixed();
    problem.Solve(30);      // 求解问题，设置最大迭代步数

    // 第八步：计算平均误差
    std::cout << "\nStep 8: Compute average residual." << std::endl;
    Scalar residual = 0.0;
    Scalar residual_ceres = 0.0;
    Scalar residual_init = 0.0;
    int cnt = 0;
    for (auto landmark : landmarkVertices) {
        GraphOptimizor::Vector3<Scalar> diff = points[cnt] - landmark->GetParameters();
        GraphOptimizor::Vector3<Scalar> diff_ceres = points[cnt] - points_esti[cnt];
        GraphOptimizor::Vector3<Scalar> diff_init = points[cnt] - noisy_points[cnt];
        residual_ceres += GraphOptimizor::ComputeTranslationMagnitude(diff_ceres);
        residual += GraphOptimizor::ComputeTranslationMagnitude(diff);
        residual_init += GraphOptimizor::ComputeTranslationMagnitude(diff_init);
        ++cnt;
    }
        
    std::cout << "  Average landmark position residual for initialization is " << residual_init / Scalar(cnt) << " m" << std::endl;
    std::cout << "  Average landmark position residual for GraphOptimizor is " << residual / Scalar(cnt) << " m" << std::endl;
    std::cout << "  Average landmark position residual for ceres solver   is " << residual_ceres / Scalar(cnt) << " m" << std::endl;

    residual = Scalar(0);
    residual_ceres = Scalar(0);
    residual_init = Scalar(0);
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        auto param = cameraVertices[i]->GetParameters();
        GraphOptimizor::Quaternion<Scalar> q_wc = GraphOptimizor::Quaternion<Scalar>(param[6], param[3], param[4], param[5]);
        GraphOptimizor::Quaternion<Scalar> diff = cameras[i].qwc.inverse() * q_wc;
        GraphOptimizor::Quaternion<Scalar> diff_ceres = cameras[i].qwc.inverse() * cameras_esti[i].qwc;
        GraphOptimizor::Quaternion<Scalar> diff_init = cameras[i].qwc.inverse() * cameras[i].qwc_noisy;
        residual += GraphOptimizor::ComputeRotationMagnitude(diff);
        residual_ceres += GraphOptimizor::ComputeRotationMagnitude(diff_ceres);
        residual_init += GraphOptimizor::ComputeRotationMagnitude(diff_init);
    }
    std::cout << "  Average camera atitude residual at initialization is  " << residual_init / Scalar(cameras.size()) << " rad, " <<
            residual_init/ Scalar(cameras.size()) * 57.29578049 << " deg" << std::endl;
    std::cout << "  Average camera atitude residual for GraphOptimizor is " << residual / Scalar(cameras.size()) << " rad, " <<
        residual / Scalar(cameras.size()) * 57.29578049 << " deg" << std::endl;
    std::cout << "  Average camera atitude residual for ceres solver is   " << residual_ceres / Scalar(cameras.size()) << " rad, " <<
            residual_ceres / Scalar(cameras.size()) * 57.29578049 << " deg" << std::endl;

    residual = Scalar(0);
    residual_ceres = Scalar(0);
    residual_init = Scalar(0);
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        auto param = cameraVertices[i]->GetParameters();
        GraphOptimizor::Vector3<Scalar> t_wc = param.template head<3>();
        GraphOptimizor::Vector3<Scalar> diff = cameras[i].twc - t_wc;
        GraphOptimizor::Vector3<Scalar> diff_ceres = cameras[i].twc - cameras_esti[i].twc;
        GraphOptimizor::Vector3<Scalar> diff_init = cameras[i].twc - cameras[i].twc_noisy;
        residual += GraphOptimizor::ComputeTranslationMagnitude(diff);
        residual_ceres += GraphOptimizor::ComputeTranslationMagnitude(diff_ceres);
        residual_init += GraphOptimizor::ComputeTranslationMagnitude(diff_init);
    }
    std::cout << "  Average camera position residual for initialization is " << residual_init / Scalar(cameras.size()) << " m" << std::endl;
    std::cout << "  Average camera position residual for GraphOptimizor is " << residual / Scalar(cameras.size()) << " m" << std::endl;
    std::cout << "  Average camera position residual for Ceres solver is   " << residual_ceres / Scalar(cameras.size()) << " m" << std::endl;



    return 0;
}



/*
Scalar = double
    delta_Xps is  0.00828004 -0.00534831 -0.00123691 -0.00106144  -0.0010599 0.000153959   0.0123405 0.000689327  0.00973075 0.000637963 -0.00212908 -0.00164504
    delta_Xl is 0.000156697  2.8801e-05 9.44395e-05
    delta_Xl is 0.000124201 5.88189e-05 0.000122222
    delta_Xl is  7.34142e-05 -1.55349e-05  4.45771e-05
    Cost [sum, mean] is 0.00355013, 0.000591689
    the norm of delta_x is 0.0193261

Scalar = float
    delta_Xps is   0.0083154  -0.0054019 -0.00122732 -0.00106937  -0.0010614 0.000150353   0.0123269 0.000655746  0.00971438 0.000634997 -0.00212309 -0.00164592
    delta_Xl is 0.000157667 2.81991e-05  9.4727e-05
    delta_Xl is 0.000124961 5.93257e-05 0.000123019
    delta_Xl is  7.39621e-05 -1.53781e-05  4.48158e-05
    Cost [sum, mean] is 0.00357319, 0.000595532
    the norm of delta_x is 0.0193402

*/