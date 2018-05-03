#ifndef SPARK_APP_SPARK_FOLLOWER_SRC_NXFOLLOWER_HPP_
#define SPARK_APP_SPARK_FOLLOWER_SRC_NXFOLLOWER_HPP_

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <sstream>
#include<iostream>
#include <vector>

#include<sensor_msgs/image_encodings.h>
#include<image_transport/image_transport.h>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>



namespace nxfollower
{

    /*----------服务socket的宏----------*/
    //待传输图像默认大小为 640*480，可修改
#define IMGWIDTH 640
#define IMGHEIGHT 480
#define PACKAGENUM 2
    //默认格式为CV_8UC3
#define BUFFERSIZE IMGWIDTH*IMGHEIGHT*3/PACKAGENUM

    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

    class NxFollowerNode
    {

        static void onMouse(int event, int x, int y, int, void *userdata)
        {
            NxFollowerNode *temp = reinterpret_cast<NxFollowerNode *>(userdata);
            temp->mouseHandle(event, x, y);
        }

    private:

        /*----------服务socket的变量----------*/
        struct sentbuf
        {
            char buf[BUFFERSIZE];
            int flag;
            int rect[4];
        };

        int sockClient;
        struct sentbuf data;
        int rectData[4];
        int addPort;

        /*----------服务CT算法的变量----------*/
        int featureMinNumRect;
        int featureMaxNumRect;
        int featureNum;
        std::vector<std::vector<cv::Rect> > features;
        std::vector<std::vector<float> > featuresWeight;
        int rOuterPositive;
        std::vector<cv::Rect> samplePositiveBox;
        std::vector<cv::Rect> sampleNegativeBox;
        int rSearchWindow;
        cv::Mat imageIntegral;
        cv::Mat samplePositiveFeatureValue;
        cv::Mat sampleNegativeFeatureValue;
        std::vector<float> muPositive;
        std::vector<float> sigmaPositive;
        std::vector<float> muNegative;
        std::vector<float> sigmaNegative;
        float learnRate;
        std::vector<cv::Rect> detectBox;
        cv::Mat detectFeatureValue;
        cv::RNG rng;
        cv::Mat lastGray;
        cv::Mat currentGray;
        int ctState;

        /*-----------窗口变量----------*/
        std::string input;
        std::string output;
        int windowSize;
        int winWidth;
        int winHeight;

        /*----------用来服务鼠标选取操作的变量----------*/
        bool isSelect;
        int isTrack;
        cv::Point startPoint;
        cv::Rect selectRect;
        cv::Mat testGlobalImg;

        /*----------用来服务计算直方图的数据变量----------*/
        // 选取矩形内的hsv图像
        cv::Mat targetImgHSV;
        // hsv空间图像的直方图
        cv::Mat dstHist;
        // 直方图中，为一维竖条的个数
        int histSize;
        // 统计的范围
        float histR[2];
        const float *histRange;
        // 用来计算直方图的通道数组
        int channels[2];

        /*----------用来服务camshift算法的变量----------*/
        cv::Mat mask, hue, hist, backproj, histImg;
        int vmin , vmax, smin;
        cv::Rect trackWindow;
        cv::Rect oldBox;
        bool isMiss;
        cv::Mat missBackProj;

        /*----------用来服务ROS的数据变量----------*/
        ros::NodeHandle nhandle;
        ros::NodeHandle pnhandle;
        // 控制命令的发布者
        ros::Publisher cmdvel_pub;
        // 点云数据的订阅者
        ros::Subscriber cloud_sub;
        // RGB图像数据的订阅者
        ros::Subscriber rgb_sub;

        double min_y_;   /**< 框中最小的y */
        double max_y_;   /**< 框中最大的y */
        double min_x_;   /**< 框中最小的x */
        double max_x_;   /**< 框中最大的x */
        double max_z_;   /**< 框中最大的z */
        // 以上变量用来设置框的大小
        double goal_z_;  /**< 离机器人的距离，以保持质心 */
        double z_scale_; /**< 移动机器人速度的缩放系数 */
        double x_scale_; /**< 旋转机器人速度的缩放系数 */
        double z_thre;
        double x_thre;
        double max_vx;                 /*x的最大速度*/
        double max_vz;                 /*z的最大速度*/
        double max_depth_, min_depth_; /**< 深度的范围 */
        double goal_depth_;            /**< 保持深度距离，以保证质心 */
        double depth_thre;
        double y_thre;

        float depthLine;
        bool robotRun;
        bool useWho;

    public:
        // .cpp文件调用的构造函数
        // 配置包裹矩形的参数和速度限制的参数
        // 立方体的长(x):0.4,宽(z):0.2,高(y):0.4
        NxFollowerNode(ros::NodeHandle nh, ros::NodeHandle pnh)
            : min_y_(0.1), max_y_(0.5), min_x_(-0.2), max_x_(0.2), max_z_(0.8), goal_z_(0.6), z_scale_(1.0), x_scale_(5.0)
        {
            nhandle = nh;
            pnhandle = pnh;

            //为什么还要再一次配置？
            min_x_ = -0.2;
            max_x_ = 0.2;
            min_y_ = -0.1;
            max_y_ = 0.3;
            max_z_ = 1.5;
            goal_z_ = 0.7;
            z_scale_ = 0.8;
            x_scale_ = 2;

            // ?
            z_thre = 0.05;
            x_thre = 0.05;
            y_thre = 0.087222222;

            // 最大旋转速度和最大前进速度
            max_vx = 0.4;
            max_vz = 0.8;

            // 最远观察深度，最近观察深度
            max_depth_ = 2;
            min_depth_ = 0.4;
            goal_depth_ = 0.9;
            depth_thre = 0.1;
            y_thre = 0.087222222;

            robotRun = false;
            useWho = true;

            // 配置发布者
            cmdvel_pub = nhandle.advertise<geometry_msgs::Twist>("/raw_cmd_vel", 1);
            // 配置订阅者，当收到消息时，调用函数pointCloudCb
            cloud_sub = nhandle.subscribe<PointCloud>("/camera/depth/points", 1, &NxFollowerNode::pointCloudCb, this);
            // 配置订阅者，当收到消息时，调用函数rgbImageCb
            rgb_sub = nhandle.subscribe("/camera/rgb/image_rect_color", 1, &NxFollowerNode::rgbImageCb, this);

            ROS_INFO("--------testPoint----------");

            selectRect = cv::Rect(0, 0, 0, 0);
            setupImgHandle();
            setupCamshift();
            setupCT();

            setupSocket();
        }

        // 析构函数
        virtual ~NxFollowerNode() {}

        /********************
            func: 配置图像显示数据
          ********************/
        void setupImgHandle()
        {
            input = "sImg";
            output = "oImg";
            // 初始化输入输出窗口
            //cv::namedWindow(input, 0);
            //cv::namedWindow(output, 0);
            //调整显示窗口的大小
            windowSize = 500;
            winWidth = 640;
            winHeight = 480;

            //cv::resizeWindow(input, winWidth, winHeight);
            //cv::resizeWindow(output, winWidth, winHeight);
            // 鼠标回调函数声明
            cv::setMouseCallback(output, onMouse, this);
            //鼠标状态
            isSelect = false;
        }

        /********************
              func: 初始化socket链接，阻塞等待
             ********************/
        void setupSocket()
        {
            addPort = 7770;
            if (socketConnect(addPort) < 0)
            {
                printf("connect error......\n");
            }
        }

        /********************
            func:配置camshift算法数据
          ********************/
        void setupCamshift()
        {
            isTrack = 0;
            histSize = 16;
            oldBox = cv::Rect(0, 0, 0, 0);
            // 统计的范围
            histR[0] = 0;
            histR[1] = 255;
            histRange = histR;
            channels[0] = 0;
            channels[1] = 1;
            //camshift
            histImg = cv::Mat::zeros(200, 300, CV_8UC3);
            isMiss = false;
            vmin = 10, vmax = 256, smin = 30;
            //创建三个滑块条，特定条件用滑块条选择不同参数能获得较好的跟踪效果
            //cv::createTrackbar( "Vmin", output, &vmin, 256, 0 );
            //cv::createTrackbar( "Vmax", output, &vmax, 256, 0 );
            //cv::createTrackbar( "Smin", output, &smin, 256, 0 );
        }

        /********************
            func: 配置CT算法数据
          ********************/
        void setupCT()
        {
            featureMinNumRect = 2;
            featureMaxNumRect = 4;  // 随机取值，2~4
            featureNum = 50;  // 分类器数量，特征池
            rOuterPositive = 4; // 正样本搜索框范围
            rSearchWindow = 25; // 负样本搜索框范围
            muPositive = std::vector<float>(featureNum, 0.0f);
            muNegative = std::vector<float>(featureNum, 0.0f);
            sigmaPositive = std::vector<float>(featureNum, 1.0f);
            sigmaNegative = std::vector<float>(featureNum, 1.0f);
            learnRate = 0.85f;  // 在线学习速率
            ctState = 1;

            //cv::createTrackbar( "feartureNum", output, &featureNum, 100, 0 );
            //cv::createTrackbar( "positive", output, &rOuterPositive, 50, 0 );
            //cv::createTrackbar( "negative", output, &rSearchWindow, 100, 0 );

        }

        /********************
          func:接收点云数据函数
          cloud:点云数据
        ********************/
        void pointCloudCb(const PointCloud::ConstPtr &cloud)
        {
            // 定义x,y,x的质心变量
            float x = 0.0;
            float y = 0.0;
            float z = 0.0;

            // 定义得到的点云的数量
            unsigned int n = 0;
            // 定义包含xyz三变量的point类型的结构体变量
            pcl::PointXYZ pt;

            // 遍历点云
            for (int kk = 0; kk < cloud->points.size(); kk++)
            {
                pt = cloud->points[kk];
                // 判断x,y,z的值是否为非数型
                if (!std::isnan(x) && !std::isnan(y) && !std::isnan(z))
                {
                    // 判断点云是否在我们检测的立方体内
                    if (-pt.y > min_y_ && -pt.y < max_y_ && pt.x < max_x_ && pt.x > min_x_ && pt.z < max_z_)
                    {
                        // 添加符合条件的点到总点计数器
                        x += pt.x;
                        z += pt.z;
                        n++;
                    }
                }
            }

            // 如果得到的符合条件的点云超过2000点
            if (n > 2000)
            {
                x /= n; // 得到x轴方向上的质心
                z /= n; // 得到y轴方向上的质心
                // 如果得到的z的质心超过立方体的宽度
                if (z > max_z_)
                {
                    // 发布停止机器人运动的消息
                    cmdvel_pub.publish(geometry_msgs::TwistPtr(new geometry_msgs::Twist()));
                    return;
                }
                // 调用发布命令函数
                // pubCmd(-x, z);
                depthLine = z;
            }
            else
            {
                // 发布停止机器人运动的消息
                cmdvel_pub.publish(geometry_msgs::TwistPtr(new geometry_msgs::Twist()));
            }
        }

        /********************
           func:接受rgb图像函数
         ********************/
        void rgbImageCb(const sensor_msgs::ImageConstPtr &rgbMsg)
        {
            ROS_INFO("--------testRGB----------");
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(rgbMsg, sensor_msgs::image_encodings::RGB8);
            }
            catch (cv_bridge::Exception &err)
            {
                ROS_ERROR("cv_bridge exception:%s", err.what());
                return ;
            }

            imageHandle(cv_ptr->image);
        }

        /********************
          func:图像总体处理函数
        ********************/
        void imageHandle(cv::Mat sImg)
        {
            cv::Mat grayImg;
            cv::Mat hsvImg;
            cv::Rect trackBox = cv::Rect(0, 0, 0, 0);
            cv::Point boxCenter;

            //cv::cvtColor(sImg,grayImg,CV_RGB2GRAY);
            testGlobalImg = sImg;

            cv::cvtColor(sImg, hsvImg, CV_RGB2HSV);

            // 如果选择框较小，则选择CT压缩感知算法
            if (useWho)
            {
                trackBox = imageCamshift(hsvImg, selectRect);
            }
            else
            {
                trackBox = imageCT(sImg, selectRect);
            }
            //trackBox = imageCamshift(hsvImg, selectRect);
            //trackBox = imageCT(sImg,selectRect);

            transmit(sImg, trackBox);

            if (getData() > 0)
            {

                switch(rectData[0])
                {
                // 关闭图片流
                case 77777:
                    //socketDisconnect();
                    socketConnect( ++addPort);
                    break;
                // 启动spark
                case 11111:
                    robotRun = true;
                    break;
                // 停止spark
                case 00000:
                    robotRun = false;
                    break;
                // 得到感兴趣的选择框
                default:
                    selectRect.x = rectData[0];
                    selectRect.y = rectData[1];
                    selectRect.width = rectData[2];
                    selectRect.height = rectData[3];
                    if (selectRect.width > 0 && selectRect.height > 0)
                    {
                        isTrack = -1;   // 置为-1表示重新计算跟踪
                        ctState = 0;
                    }
                    if (selectRect.width * 4 > winWidth || selectRect.height * 4 > winHeight )
                    {
                        useWho = true;
                    }
                    else
                    {
                        useWho = false;
                    }
                    break;
                }
            }

            // 判断鼠标是否已经选取了roi
            // 绘图，控制spark运动
            if (selectRect.width > 0 && selectRect.height > 0 && !isMiss && robotRun)
            {
                //cv::rectangle(sImg, trackBox, cv::Scalar(255, 0, 0), 2);
                boxCenter = getRectCenter(trackBox);
                // 发布运动广播
                float angle = angleHandle(boxCenter.x);
                float linear = linearHandle();
                publishHandle(linear, angle);
            }

            //cv::imshow(input, sImg);
            //cv::imshow(output, hsvImg);
            cv::waitKey(5);
        }

        /********************
          func:图像噪点过滤函数
        ********************/
        void imageFilter(cv::Mat img)
        {

        }

        /********************
          func:使用camshifth追踪算法
          img:为hsv空间的图像，3通道
        ********************/
        cv::Rect imageCamshift(cv::Mat hsvImg, cv::Rect histRect)
        {
            //cv::Mat tempImg;( hsvImg.size(),hsvImg.depth(), cv::Scalar(255,255,0) );
            if (isTrack)
            {

                int _vmin = vmin, _vmax = vmax;
                cv::inRange(hsvImg, cv::Scalar(0, smin, MIN(_vmin, _vmax)), cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                hue.create(hsvImg.size(), hsvImg.depth());
                //tempImg = cv::Mat::zeros(hsvImg.size(),hsvImg.depth());


                // 输入的矩阵的某些通道拆分复制给对应的输出矩阵的某些通道中
                // 分离H分量
                int ch[] = {0, 0};
                cv::mixChannels(&hsvImg, 1, &hue, 1, ch, 1);
                // int testCH[] = {0,0,1,1};
                // cv::mixChannels(&hsvImg,1,&hue,1,testCH,2);

                if (isTrack < 0)
                {
                    cv::Mat roi(hue, histRect), maskRoi(mask, histRect);
                    // int histChan[] = {0,1};
                    // calcHist(&roi, 1, histChan, maskRoi,hist, 2, &histSize, &histRange,true, false );
                    cv::calcHist(&roi, 1, 0, maskRoi, hist, 1, &histSize, &histRange);
                    cv::normalize(hist, hist, 0, 255, CV_MINMAX);
                    trackWindow = histRect;
                    isTrack = 1;
                }

                // 根据直方图hist计算整幅图像的反向投影图backproj,backproj与hue相同大小
                cv::calcBackProject(&hue, 1, 0, hist, backproj, &histRange);
                backproj &= mask;

                if (!isMiss)
                {
                    // 人为修改反向投影图，使得周围环境对被检查物的干扰最小
                    // 算子
                    cv::Mat elementE = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                    cv::Mat elementD = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

                    // 取得形态学处理的小范围
                    cv::Point centerPoint = getRectCenter(trackWindow);
                    int targetX = (centerPoint.x - selectRect.width / 2) < 0 ? 0 : (centerPoint.x - selectRect.width / 2);
                    int targetY =  (centerPoint.y - selectRect.height / 2) < 0 ? 0 : (centerPoint.y - selectRect.height / 2);
                    int targetW = (targetX + selectRect.width) > winWidth ? trackWindow.width : selectRect.width;
                    int targetH = (targetY + selectRect.height) > winHeight ? trackWindow.height : selectRect.height;

                    cv::Rect shapeRect = cv::Rect(targetX, targetY, targetW, targetH);
                    cv::erode(backproj, backproj, elementE);
                    cv::dilate(backproj(shapeRect), backproj(shapeRect), elementD);
                    cv::dilate(backproj(shapeRect), backproj(shapeRect), elementD);
                    // 腐蚀操作
                    // cv::erode( backproj, backproj, elementE );
                    // // 膨胀操作
                    // cv::dilate(backproj(trackWindow), backproj(trackWindow), elementD);
                    // cv::dilate(backproj(trackWindow), backproj(trackWindow), elementD);
                    /*
                    cv::Rect roiRect = isTrack==1?trackWindow:selectRect;
                    cv::imshow("hue0",hue);
                    cv::addWeighted(tempImg, 0.7, hue, 0.3, 0.0, tempImg);
                    hue = hue(roiRect);
                    hue.copyTo(tempImg(roiRect));
                    hue = tempImg.clone();
                    cv::imshow("hue",hue);*/
                }
                else
                {
                    //backproj = missBackProj.clone();
                    isMiss = false;
                }

                //cv::imshow("333",backproj);
                //cv::imshow("hsv",hsvImg);
                // trackWindow传入的是自身的引用，故一直在变化
                // 模板类，item1:类型;item2:迭代的最大次数;item3:特定的阈值
                // cv::RotatedRect trackBox =
                cv::CamShift(backproj, trackWindow, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));

                //std::string text = int2Str(trackWindow.width);
                //cv::putText(testGlobalImg,text,cv::Point(50,60),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);

                // 当跟踪框几乎消失时，人为的添加,防止崩溃
                if (trackWindow.area() <= 5)
                {
                    // 消失在左边还是右边
                    if (trackWindow.x < (winWidth / 2))
                    {
                        trackWindow = cv::Rect(0, 0, 100, winHeight);
                    }
                    else
                    {

                        trackWindow = cv::Rect(winWidth - 100, 0, 100, winHeight);
                    }
                    isMiss = true;
                    //missBackProj = backproj.clone();
                }

                return trackWindow;//trackBox.boundingRect();
            }
        }

        /********************
            func: 使用光流金字塔+camshift追踪算法
          ********************/
        void imageCamFlower()
        {
        }

        /********************
            func: 使用压缩感知跟踪算法
            sImg: 传入图像
          ********************/
        cv::Rect imageCT(cv::Mat sImg, cv::Rect &ctRect)
        {
            switch (ctState)
            {
            case 1:
                cv::cvtColor(sImg, lastGray, CV_BGR2GRAY);
                break;
            case 0:
                CTinit(lastGray, ctRect);
                ctState = -1;
                break;
            case -1:
                cv::cvtColor(sImg, currentGray, CV_BGR2GRAY);
                CTprocessFrame(currentGray, ctRect);
                break;
            default:
                break;
            }
            return ctRect;
        }

        /********************
            func:跟踪算法失效后的矫正函数
          ********************/
        void imageFix()
        {

        }

        /********************
           func:鼠标回调函数
        ********************/
        void mouseHandle(int event, int x, int y)
        {
            if (isSelect)
            {
                selectRect.x = MIN(x, startPoint.x);
                selectRect.y = MIN(y, startPoint.y);
                selectRect.width = std::abs(x - startPoint.x);
                selectRect.height = std::abs(y - startPoint.y);
            }

            switch(event)
            {
            case CV_EVENT_LBUTTONDOWN:
                startPoint = cv::Point(x, y);
                selectRect = cv::Rect(x, y, 0, 0);
                isSelect = true;
                ctState = 1;
                break;
            case CV_EVENT_LBUTTONUP:
                isSelect = false;
                if (selectRect.width > 0 && selectRect.height > 0)
                {
                    isTrack = -1;   // 置为-1表示重新计算跟踪
                }
                ctState = 0;
                break;
            }
        }

        /********************
          func:前后运动控制函数
         ********************/
        float linearHandle()
        {
            // 线速度，角速度，二者的变化系数
            float xLinear = 0;
            float xScale = 1.2;
            if (depthLine != 10000)
            {
                // 线速度 = (当前质心深度-目标深度)*缩放系数，控制前进
                xLinear = (depthLine - goal_depth_) * xScale;
                // 阈值分割判断最小修改
                if (depth_thre > fabs(depthLine - goal_depth_))
                {
                    xLinear = 0;
                }
            }
            return xLinear;
        }

        /********************
          func:旋转运动控制函数
         ********************/
        float angleHandle(int pointX)
        {
            float zAngular = 0;
            float zScale = -1.8;
            float centerPoint = winWidth / 2;
            float disX = pointX - centerPoint;
            int disThre = 12;
            if (disThre > disX && disX > -disThre)
            {
                zAngular = 0;
            }
            else
            {
                zAngular = atan2(disX, centerPoint) * zScale;
            }

            return zAngular;
        }

        /********************
          func:发送控制消息函数
        ********************/
        void publishHandle(float xLinear, float zAngular)
        {
            // 创建一个消息体
            geometry_msgs::TwistPtr cmd(new geometry_msgs::Twist());
            // 赋值直线速度
            cmd->linear.x = xLinear;
            // 赋值角速度
            cmd->angular.z = zAngular;
            // 发布消息
            cmdvel_pub.publish(cmd);
        }

        /********************
          func:取得方框中心点函数
        ********************/
        cv::Point getRectCenter(cv::Rect box)
        {
            int centerX = box.x + box.width / 2;
            int centerY = box.y + box.height / 2;
            return cv::Point(centerX, centerY);
        }

        /********************
          func:int类型转string类型函数
         ********************/
        std::string int2Str(int data)
        {
            std::stringstream ss;
            std::string str;
            ss << data;
            ss >> str;
            return str;
        }

        /*------------------------------------------Socket------------------------------------------*/

        /********************
          func: 建立socket链接
          port: 端口
             ********************/
        int socketConnect(int PORT)
        {
            int socketFd = socket(AF_INET, SOCK_STREAM, 0);

            struct sockaddr_in serverAddr;
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_port = htons(PORT);
            serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

            if(bind(socketFd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1)
            {
                perror("bind");
                return -1;
            }

            if(listen(socketFd, 5) == -1)
            {
                perror("listen");
                return -1;
            }

            struct sockaddr_in clientAddr;
            socklen_t length = sizeof(clientAddr);

            sockClient = accept(socketFd, (struct sockaddr *)&clientAddr, &length);
            if(sockClient < 0)
            {
                perror("connect");
                return -1;
            }
            else
            {
                printf("connect successful!\n");
                return 1;
            }

            close(socketFd);
        }

        /********************
              func: 断开socket链接
             ********************/
        void socketDisconnect(void)
        {
            close(sockClient);
        }

        /********************
              func: 建立socket链接
          image: 需要传输的图片
          roiRect: 感兴趣的区域
             ********************/
        int transmit(cv::Mat image, cv::Rect roiRect)
        {
            if (image.empty())
            {
                printf("empty image\n\n");
                return -1;
            }
            else
            {
                cv::resize(image, image, cv::Size(IMGWIDTH, IMGHEIGHT));
            }

            if(image.cols != IMGWIDTH || image.rows != IMGHEIGHT || image.type() != CV_8UC3)
            {
                printf("the image must satisfy : cols == IMGWIDTH（%d）  rows == IMGHEIGHT（%d） type == CV_8UC3\n\n", IMGWIDTH, IMGHEIGHT);
                return -1;
            }

            data.rect[0] = roiRect.x;
            data.rect[1] = roiRect.y;
            data.rect[2] = roiRect.width;
            data.rect[3] = roiRect.height;

            // 分为两次发送
            for(int k = 0; k < PACKAGENUM; k++)
            {
                int num1 = IMGHEIGHT / PACKAGENUM * k;
                for (int i = 0; i < IMGHEIGHT / PACKAGENUM; i++)
                {
                    int num2 = i * IMGWIDTH * 3;
                    uchar *ucdata = image.ptr<uchar>(i + num1);
                    for (int j = 0; j < IMGWIDTH * 3; j++)
                    {
                        data.buf[num2 + j] = ucdata[j];
                    }
                }

                if(k == PACKAGENUM - 1)
                    data.flag = 2;
                else
                    data.flag = 1;

                if (send(sockClient, (char *)(&data), sizeof(data), 0) < 0)
                {
                    printf("send image error: %s(errno: %d)\n", strerror(errno), errno);
                    return -1;
                }
            }
        }

        /********************
              func: 得到上位机传来的数据
             ********************/
        int getData()
        {

            memset(&rectData, 0, sizeof(rectData));

            int len =  recv(sockClient, rectData, sizeof(rectData), MSG_DONTWAIT);
            if (len < 0)
            {
                return -1;
            }
            return 1;
        }

        /*------------------------------------------Socket------------------------------------------*/

        /*------------------------------------------CT-------------------------------------------*/
        /********************
          func:计算haar特征
          _objectBox:roi的矩形
          _numFeature:特征池50
        ********************/
        void CThaarFeature(cv::Rect &_objectBox, int _numFeature)
        {
            // 两个全局变量，表示特征，稀疏随机矩阵
            features = std::vector<std::vector<cv::Rect> >(_numFeature, std::vector<cv::Rect>());
            featuresWeight = std::vector<std::vector<float> >(_numFeature, std::vector<float>());

            int numRect;
            cv::Rect rectTemp;
            float weightTemp;

            for (int i = 0; i < _numFeature; i++)
            {
                // 产生范围内的随机数2~4
                numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));

                for (int j = 0; j < numRect; j++)
                {
                    // 在选取的roi中随机取得n个小矩形
                    rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
                    rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
                    rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
                    rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
                    features[i].push_back(rectTemp);

                    // 取得随机的特征权重
                    weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
                    featuresWeight[i].push_back(weightTemp);

                }
            }
        }

        /********************
         func: 计算正、负样本图像模板的坐标
         _image: 图像帧
         _objectBox: 选择的roi矩形
         _rInner: 内部采样半径
         _rOuter: 外部采用半径
         _maxSampleNum: 最大采样数
         _ sampleBox: 采样后的图像的坐标，引用返回
        ********************/
        void CTsampleRect(cv::Mat &_image, cv::Rect &_objectBox, float _rInner, float _rOuter, int _maxSampleNum, std::vector<cv::Rect> &_sampleBox)
        {
            int rowsz = _image.rows - _objectBox.height - 1;
            int colsz = _image.cols - _objectBox.width - 1;
            float inradsq = _rInner * _rInner;
            float outradsq = _rOuter * _rOuter;

            int dist;

            int minrow = MAX(0, (int)_objectBox.y - (int)_rInner);
            int maxrow = MIN((int)rowsz - 1, (int)_objectBox.y + (int)_rInner);
            int mincol = MAX(0, (int)_objectBox.x - (int)_rInner);
            int maxcol = MIN((int)colsz - 1, (int)_objectBox.x + (int)_rInner);

            int i = 0;

            // 取得概率
            float prob = ((float)(_maxSampleNum)) / (maxrow - minrow + 1) / (maxcol - mincol + 1);

            int r;
            int c;

            _sampleBox.clear();//important
            cv::Rect rec(0, 0, 0, 0);

            for( r = minrow; r <= (int)maxrow; r++ )
            {
                for( c = mincol; c <= (int)maxcol; c++ )
                {
                    dist = (_objectBox.y - r) * (_objectBox.y - r) + (_objectBox.x - c) * (_objectBox.x - c);

                    if( rng.uniform(0., 1.) < prob && dist < inradsq && dist >= outradsq )
                    {
                        rec.x = c;
                        rec.y = r;
                        rec.width = _objectBox.width;
                        rec.height = _objectBox.height;

                        _sampleBox.push_back(rec);
                        i++;
                    }
                }
            }
            _sampleBox.resize(i);
        }

        /********************
            func: 在检测对象时计算样本坐标
            _image:图片
            _objectBox:选取的roi矩形
            _srw: 窗口值
            _sampleBox:样本矩形
        ********************/
        void CTsampleRect(cv::Mat &_image, cv::Rect &_objectBox, float _srw, std::vector<cv::Rect> &_sampleBox)
        {
            int rowsz = _image.rows - _objectBox.height - 1;
            int colsz = _image.cols - _objectBox.width - 1;
            float inradsq = _srw * _srw;

            int dist;

            int minrow = MAX(0, (int)_objectBox.y - (int)_srw);
            int maxrow = MIN((int)rowsz - 1, (int)_objectBox.y + (int)_srw);
            int mincol = MAX(0, (int)_objectBox.x - (int)_srw);
            int maxcol = MIN((int)colsz - 1, (int)_objectBox.x + (int)_srw);

            int i = 0;

            int r;
            int c;

            cv::Rect rec(0, 0, 0, 0);
            _sampleBox.clear();//important

            for( r = minrow; r <= (int)maxrow; r++ )
            {
                for( c = mincol; c <= (int)maxcol; c++ )
                {
                    dist = (_objectBox.y - r) * (_objectBox.y - r) + (_objectBox.x - c) * (_objectBox.x - c);

                    if( dist < inradsq )
                    {

                        rec.x = c;
                        rec.y = r;
                        rec.width = _objectBox.width;
                        rec.height = _objectBox.height;

                        _sampleBox.push_back(rec);

                        i++;
                    }
                }
            }
            _sampleBox.resize(i);

        }

        /********************
            func: 计算样本的特征值
            _imageIntegral: 当前图片流的图片的积分图
            _sampleBox: 样本图的坐标
            _sampleFeatureValue:样本图的特征值
        ********************/
        void CTgetFeatureValue(cv::Mat &_imageIntegral, std::vector<cv::Rect> &_sampleBox, cv::Mat &_sampleFeatureValue)
        {
            int sampleBoxSize = _sampleBox.size();
            // 创建一张空图，降维后的矩阵
            _sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
            float tempValue;
            int xMin;
            int xMax;
            int yMin;
            int yMax;

            for (int i = 0; i < featureNum; i++)
            {
                for (int j = 0; j < sampleBoxSize; j++)
                {
                    tempValue = 0.0f;
                    for (size_t k = 0; k < features[i].size(); k++)
                    {
                        // 使用积分图可以高效的计算出v(即降维后的图)
                        xMin = _sampleBox[j].x + features[i][k].x;
                        xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
                        yMin = _sampleBox[j].y + features[i][k].y;
                        yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;
                        tempValue += featuresWeight[i][k] *
                                     (_imageIntegral.at<float>(yMin, xMin) +
                                      _imageIntegral.at<float>(yMax, xMax) -
                                      _imageIntegral.at<float>(yMin, xMax) -
                                      _imageIntegral.at<float>(yMax, xMin));
                    }
                    _sampleFeatureValue.at<float>(i, j) = tempValue;
                }
            }
        }

        /********************
            func: 更新高斯分类器的均值和方差
            _smapleFeatureValue: 样本的特征值
            _mu: 参数mu
            _sigma: 参数sigma
            _learnRate: 学习的速率
        ********************/
        void CTclassifierUpdate(cv::Mat &_sampleFeatureValue, std::vector<float> &_mu, std::vector<float> &_sigma, float _learnRate)
        {
            cv::Scalar muTemp;
            cv::Scalar sigmaTemp;

            for (int i = 0; i < featureNum; i++)
            {
                cv::meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
                // 第六个方程
                _sigma[i] = (float)sqrt( _learnRate * _sigma[i] * _sigma[i] + (1.0f - _learnRate) * sigmaTemp.val[0] * sigmaTemp.val[0]
                                         + _learnRate * (1.0f - _learnRate) * (_mu[i] - muTemp.val[0]) * (_mu[i] - muTemp.val[0]));
                // 第六个方程
                _mu[i] = _mu[i] * _learnRate + (1.0f - _learnRate) * muTemp.val[0];
            }
        }

        /********************
            func:计算分类器系数
            ----
        ********************/
        void CTradioClassifier(std::vector<float> &_muPos, std::vector<float> &_sigmaPos,
                               std::vector<float> &_muNeg, std::vector<float> &_sigmaNeg,
                               cv::Mat &_sampleFeatureValue, float &_radioMax, int &_radioMaxIndex)
        {
            float sumRadio;
            _radioMax = -FLT_MAX;
            _radioMaxIndex = 0;
            float pPos;
            float pNeg;
            int sampleBoxNum = _sampleFeatureValue.cols;

            for (int j = 0; j < sampleBoxNum; j++)
            {
                sumRadio = 0.0f;
                for (int i = 0; i < featureNum; i++)
                {
                    pPos = exp( (_sampleFeatureValue.at<float>(i, j) - _muPos[i]) * (_sampleFeatureValue.at<float>(i, j) - _muPos[i]) / -(2.0f * _sigmaPos[i] * _sigmaPos[i] + 1e-30) ) / (_sigmaPos[i] + 1e-30);
                    pNeg = exp( (_sampleFeatureValue.at<float>(i, j) - _muNeg[i]) * (_sampleFeatureValue.at<float>(i, j) - _muNeg[i]) / -(2.0f * _sigmaNeg[i] * _sigmaNeg[i] + 1e-30) ) / (_sigmaNeg[i] + 1e-30);
                    // 第四个方程
                    sumRadio += log(pPos + 1e-30) - log(pNeg + 1e-30);
                }
                if (_radioMax < sumRadio)
                {
                    _radioMax = sumRadio;
                    _radioMaxIndex = j;
                }
            }
        }

        /********************
            func:CT初始化
            _frame:选取roi时的最后一帧的灰度图
            _objectBox:选取的roi的矩形信息（左下角为x,y的起点）
        ********************/
        void CTinit(cv::Mat &_frame, cv::Rect &_objectBox)
        {
            // 得到haar特征
            CThaarFeature(_objectBox, featureNum);

            // 得到正负样本的位置
            CTsampleRect(_frame, _objectBox, rOuterPositive, 0, 1000000, samplePositiveBox);
            CTsampleRect(_frame, _objectBox, rSearchWindow * 1.5, rOuterPositive + 4.0, 100, sampleNegativeBox);
            // 计算出一张积分图
            cv::integral(_frame, imageIntegral, CV_32F);
            // 得到正负样本的特征值
            CTgetFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
            CTgetFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
            // 更新正负样本的高斯分类器的均值和方差
            CTclassifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
            CTclassifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
        }

        /********************
            func: 对每一帧的图像进行跟踪
        ********************/
        void CTprocessFrame(cv::Mat &_frame, cv::Rect &_objectBox)
        {
            // 预测
            CTsampleRect(_frame, _objectBox, rSearchWindow, detectBox);
            cv::integral(_frame, imageIntegral, CV_32F);
            // 把当前帧降维
            CTgetFeatureValue(imageIntegral, detectBox, detectFeatureValue);
            int radioMaxIndex;
            float radioMax;
            CTradioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
            // 找出目标并赋值给原roi用来更新
            _objectBox = detectBox[radioMaxIndex];

            // update
            CTsampleRect(_frame, _objectBox, rOuterPositive, 0.0, 1000000, samplePositiveBox);
            CTsampleRect(_frame, _objectBox, rSearchWindow * 1.5, rOuterPositive + 4.0, 100, sampleNegativeBox);
            // 得到正负样本的特征值
            CTgetFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
            CTgetFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
            // 更新正负样本的高斯分类器的均值和方差
            CTclassifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
            CTclassifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
        }
        /*--------------------------------------------CT------------------------------------------*/


        void spin()
        {
            ros::spin();
        }


    };
}
#endif
