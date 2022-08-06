#include <iostream>
#include <random>

#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <vertex_pose.hpp>
#include <vertex_landmark_pos.hpp>
#include <edge_reprojection_pos.hpp>

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
    int featureNums = 100;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 10;      // 相机数目

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

    // 第三步：构造相机 pose 节点，并添加到 problem 中
    std::cout << "\nStep 3: Add camera pose vertices." << std::endl;
    std::vector<std::shared_ptr<GraphOptimizor::VertexPose<Scalar>>> cameraVertices;
    for (size_t i = 0; i < cameras.size(); ++i) {
        GraphOptimizor::VectorX<Scalar> param(7);
        param << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w();
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
    std::default_random_engine generator;
    std::normal_distribution<double> landmark_position_noise(0.0, 0.5);
    for (size_t i = 0; i < points.size(); ++i) {
        // 构造特征点的节点
        GraphOptimizor::VectorX<Scalar> param = points[i];
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
            std::shared_ptr<GraphOptimizor::HuberKernel<Scalar>> kernel(new GraphOptimizor::HuberKernel<Scalar>(1.0));
            edge->SetKernel(kernel);
            problem.AddEdge(edge);
        }
    }
    std::cout << "Add " << landmarkVertices.size() << " landmark vertices, problem has " << problem.GetVertexNum() << " vertices and " <<
        problem.GetEdgeNum() << " edges." << std::endl;

    // 第五步：打印出优化求解的初值
    std::cout << "\nStep 5: Show the initial value of parameters." << std::endl;
    std::cout << "=================== Initial parameters -> Landmark Position ===================" << std::endl;
    size_t i = 0;
    for (auto landmark : landmarkVertices) {
        std::cout << "  id " << i << " : gt [" << points[i].transpose() << "],\top [" << landmark->GetParameters().transpose() << "]" << std::endl;
        ++i;
        if (i > 10) {
            break;
        }
    }
    std::cout << "=================== Initial parameters -> Camera Rotation ===================" << std::endl;
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        std::cout << "  id " << i << " : gt [" << cameras[i].qwc.w() << ", " << cameras[i].qwc.x() << ", " << cameras[i].qwc.y() << ", " << cameras[i].qwc.z() <<
            "],\top [" << cameraVertices[i]->GetParameters()(6) << ", " << cameraVertices[i]->GetParameters()(3) << ", " << cameraVertices[i]->GetParameters()(4) <<
            ", " << cameraVertices[i]->GetParameters()(5) << "]" << std::endl;
    }
    std::cout << "=================== Initial parameters -> Camera Position ===================" << std::endl;
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        std::cout << "  id " << i << " : gt [" << cameras[i].twc.transpose() << "],\top [" << cameraVertices[i]->GetParameters().head<3>().transpose() << "]" << std::endl;
    }

    // 第六步：配置相关参数，求解问题
    std::cout << "\nStep 6: Start solve problem." << std::endl;
    problem.SetMargnedVertexTypesWhenSolving(type_landmarkVertex);  // 设置在求解过程中需要暂时被边缘化的节点的类型
    problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::LM_Auto); // 设置数值优化方法
    // problem.SetMethod(GraphOptimizor::Problem<Scalar>::Method::DogLeg);
    problem.SetLinearSolver(GraphOptimizor::Problem<Scalar>::LinearSolver::PCG_Solver);
    // problem.LM_SetDampParameter(11.0, 10.0);    // 设置 LM 算法的阻尼因子调整参数
    cameraVertices[0]->SetFixed();              // 因为是 VO 问题，固定前两帧相机 pose
    cameraVertices[1]->SetFixed();
    problem.Solve(30);      // 求解问题，设置最大迭代步数

    // 第七步：提取出求解结果，对比真值
    std::cout << "\nStep 7: Compare optimization result with ground truth." << std::endl;
    std::cout << "=================== Optimization result -> Landmark Position ===================" << std::endl;
    size_t cnt = 0;
    for (auto landmark : landmarkVertices) {
        std::cout << "  id " << cnt << " : gt [" << points[cnt].transpose() << "],\top [" << landmark->GetParameters().transpose() << "]" << std::endl;
        ++cnt;
        if (cnt > 10) {
            break;
        }
    }
    std::cout << "=================== Optimization result -> Camera Rotation ===================" << std::endl;
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        std::cout << "  id " << i << " : gt [" << cameras[i].qwc.w() << ", " << cameras[i].qwc.x() << ", " << cameras[i].qwc.y() << ", " << cameras[i].qwc.z() <<
            "],\top [" << cameraVertices[i]->GetParameters()(6) << ", " << cameraVertices[i]->GetParameters()(3) << ", " << cameraVertices[i]->GetParameters()(4) <<
            ", " << cameraVertices[i]->GetParameters()(5) << "]" << std::endl;
    }
    std::cout << "=================== Optimization result -> Camera Position ===================" << std::endl;
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        std::cout << "  id " << i << " : gt [" << cameras[i].twc.transpose() << "],\top [" << cameraVertices[i]->GetParameters().head<3>().transpose() << "]" << std::endl;
    }

    // 第八步：计算平均误差
    std::cout << "\nStep 8: Compute average residual." << std::endl;
    Scalar residual = 0.0;
    cnt = 0;
    for (auto landmark : landmarkVertices) {
        GraphOptimizor::Vector3<Scalar> diff = points[cnt] - landmark->GetParameters();
        residual += GraphOptimizor::ComputeTranslationMagnitude(diff);
        ++cnt;
    }
    std::cout << "  Average landmark position residual is " << residual / Scalar(cnt) << " m" << std::endl;

    residual = Scalar(0);
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        GraphOptimizor::VectorX<Scalar> param = cameraVertices[i]->GetParameters();
        GraphOptimizor::Quaternion<Scalar> diff = cameras[i].qwc.inverse() * GraphOptimizor::Quaternion<Scalar>(param[6], param[3], param[4], param[5]);
        residual += GraphOptimizor::ComputeRotationMagnitude(diff);
    }
    std::cout << "  Average camera atitude residual is " << residual / Scalar(cameras.size()) << " rad, " <<
        residual / Scalar(cameras.size()) * 57.29578049 << " deg" << std::endl;

    residual = Scalar(0);
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        GraphOptimizor::Vector3<Scalar> diff = cameras[i].twc - cameraVertices[i]->GetParameters().head<3>();
        residual += GraphOptimizor::ComputeTranslationMagnitude(diff);
    }
    std::cout << "  Average camera position residual is " << residual / Scalar(cameras.size()) << " m" << std::endl;

    return 0;
}