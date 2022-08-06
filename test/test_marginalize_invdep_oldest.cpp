#include <iostream>
#include <random>

#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <vertex_pose.hpp>
#include <vertex_landmark_invdep.hpp>
#include <edge_reprojection_invdep.hpp>

using Scalar = float;      // 方便切换浮点数精度

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(GraphOptimizor::Matrix3<Scalar> R, GraphOptimizor::Vector3<Scalar> t) : Rwc(R), qwc(R), twc(t) {};
    GraphOptimizor::Matrix3<Scalar> Rwc;
    GraphOptimizor::Quaternion<Scalar> qwc;
    GraphOptimizor::Vector3<Scalar> twc;

    std::unordered_map<int, GraphOptimizor::Vector3<Scalar>> featurePerId; // 该帧观测到的特征以及特征id
};


/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(std::vector<Frame> &cameraPoses, std::vector<GraphOptimizor::Vector3<Scalar>> &points) {
    int featureNums = 10;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;      // 相机数目

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





/* 程序主函数入口 */
int main() {
    std::cout << "Test GraphOptimizor solver on Mono BA problem." << std::endl;

    // 第一步：准备测试数据
    std::cout << "\nStep 1: Prepare dataset." << std::endl;
    std::vector<Frame> cameras;
    std::vector<GraphOptimizor::Vector3<Scalar>> points;
    GetSimDataInWordFrame(cameras, points);

    // 第二步：构造待求解问题
    std::cout << "\nStep 2: Construct GraphOptimizor solver." << std::endl;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_cameraVertex = 0;
    size_t type_landmarkVertex = 1;
    size_t priorSize = 0;

    // 第三步：构造相机 pose 节点，并添加到 problem 中
    std::cout << "\nStep 3: Add camera pose vertices." << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> camera_rotation_noise(0., 0.01);
    std::normal_distribution<double> camera_position_noise(0., 0.01);
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
        GraphOptimizor::Quaternion<Scalar> temp_q_wc = cameras[i].qwc * noise_q_wc;
        GraphOptimizor::Vector3<Scalar> temp_t_wc = cameras[i].twc + noise_t;

        GraphOptimizor::VectorX<Scalar> param(7);
        param << temp_t_wc, temp_q_wc.x(), temp_q_wc.y(), temp_q_wc.z(), temp_q_wc.w();
        std::shared_ptr<GraphOptimizor::VertexPose<Scalar>> cameraVertex(new GraphOptimizor::VertexPose<Scalar>());
        cameraVertex->SetParameters(param);
        cameraVertex->SetType(type_cameraVertex);
        cameraVertices.emplace_back(cameraVertex);
        problem.AddVertex(cameraVertex);
        priorSize += cameraVertex->GetCalculationDimension();
    }
    std::cout << "Add " << cameraVertices.size() << " camera vertices, problem has " << problem.GetVertexNum() << " vertices." << std::endl;

    // 第四步：构造特征点 invdep 节点，同时构造 reprojection 边，添加到 problem 中
    std::cout << "\nStep 4: Add landmark invdep vertices." << std::endl;
    std::vector<std::shared_ptr<GraphOptimizor::VertexLandmarkInvDepth<Scalar>>> landmarkVertices;
    std::normal_distribution<double> landmark_invdep_noise(0.0, 0.01);
    for (size_t i = 0; i < points.size(); ++i) {
        // 构造特征点的节点
        GraphOptimizor::Vector3<Scalar> pc = cameras[0].Rwc.transpose() * (points[i] - cameras[0].twc);
        GraphOptimizor::VectorX<Scalar> param = GraphOptimizor::Vector1<Scalar>(1.0 / (pc.z() + landmark_invdep_noise(generator)));
        std::shared_ptr<GraphOptimizor::VertexLandmarkInvDepth<Scalar>> landmarkVertex(new GraphOptimizor::VertexLandmarkInvDepth<Scalar>());
        landmarkVertex->SetParameters(param);
        landmarkVertex->SetType(type_landmarkVertex);
        landmarkVertices.emplace_back(landmarkVertex);
        problem.AddVertex(landmarkVertex);

        // 遍历此点的观测，构造 reprojection 边，添加到 problem 中
        for (size_t j = 1; j < cameras.size(); ++j) {
            GraphOptimizor::Vector2<Scalar> norm_i = cameras[0].featurePerId.find(i)->second.head<2>();
            GraphOptimizor::Vector2<Scalar> norm_j = cameras[j].featurePerId.find(i)->second.head<2>();
            std::shared_ptr<GraphOptimizor::EdgeReprojectionInvdep<Scalar>> edge(new GraphOptimizor::EdgeReprojectionInvdep<Scalar>(norm_i, norm_j));
            edge->AddVertex(landmarkVertex, 0);
            edge->AddVertex(cameraVertices[0], 1);
            edge->AddVertex(cameraVertices[j], 2);
            std::shared_ptr<GraphOptimizor::HuberKernel<Scalar>> kernel(new GraphOptimizor::HuberKernel<Scalar>(1.0));
            edge->SetKernel(kernel);
            problem.AddEdge(edge);
        }
    }
    std::cout << "Add " << landmarkVertices.size() << " landmark vertices, problem has " << problem.GetVertexNum() << " vertices and " <<
        problem.GetEdgeNum() << " edges." << std::endl;

    // 第六步：配置相关参数，边缘化最旧帧
    std::cout << "\nStep 6: Start marginalization." << std::endl;
    problem.SetMargnedVertexTypesWhenSolving(type_landmarkVertex);  // 设置在求解过程中需要暂时被边缘化的节点的类型
    std::vector<std::shared_ptr<GraphOptimizor::VertexBase<Scalar>>> needMarg;
    needMarg.emplace_back(cameraVertices.front());
    problem.Marginalize(needMarg, priorSize - cameraVertices.front()->GetCalculationDimension());

    // 第七步：打印出先验信息
    std::cout << "\nStep 7: Show prior result." << std::endl;
    GraphOptimizor::MatrixX<Scalar> prior_H, prior_JTinv;
    GraphOptimizor::VectorX<Scalar> prior_b, prior_r;
    problem.GetPrior(prior_H, prior_b, prior_JTinv, prior_r);
    std::cout << "prior H is\n" << prior_H << std::endl;
    std::cout << "prior b is\n" << prior_b << std::endl;
    std::cout << "prior JTinv is\n" << prior_JTinv << std::endl;
    std::cout << "prior r is\n" << prior_r << std::endl;

    return 0;
}