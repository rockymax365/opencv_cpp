#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<queue>
#include<string>

using namespace std;
using namespace cv;

//骨架提取
void skeletonExtraction(Mat & img)
{
	threshold(img, img, 127, 255, THRESH_BINARY);

	//临时图像
	Mat skel(img.size(), CV_8UC1, Scalar(0));
	Mat temp(img.size(), CV_8UC1);
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));
	bool done = false;

	do {
		//开操作-确保去掉小的干扰块
		morphologyEx(img, temp, MORPH_OPEN, element);
		//取反操作
		bitwise_not(temp, temp);
		//得到与源图像不同
		bitwise_and(img, temp, temp);
		//使用它提取骨架，得到是仅仅比源图像小一个像素
		bitwise_or(skel, temp, skel);
		//每次循环腐蚀，通过不断腐蚀的方式得到骨架
		erode(img, img, element);

		//对腐蚀之后的图像寻找最大值，如果被完全腐蚀则说明
		//只剩下背景黑色、已经得到骨架，退出循环
		double max;
		minMaxLoc(img, NULL, &max);
		done = (0 == max);

	} while (!done);

	imshow("s", skel);

	return;
}



#if 0
int main()
{
	int arr[10] = { 2,5,7,4,3,4,5,9,6,5 };

	findTheSmallestNumber(arr, 10, 5);

	Mat img = imread("F:/c++/test_opencv/test_opencv/img/Skeleton.JPG", IMREAD_GRAYSCALE);

	skeletonExtraction(img);

	//打开一个默认的相机
	VideoCapture capture(0);
	//检查是否成功打开
	if (!capture.isOpened())
		return -1;


	while (1)
	{
		Mat frame;
		capture >> frame;//从相机读取新一帧
		//cvtColor(frame, edges, COLOR_BGR2GRAY);//变为灰度图
		resize(frame, frame, Size(frame.cols, frame.rows));
		imshow("image", frame);
		if (waitKey(10) == 'q')
			break;
	}
	return 0;

}
#endif


//读取显示图像
void showImg(string imgPath)
{
	cv::Mat src = imread(imgPath); //B,G,R  后加IMREAD_GRAYSCALE为显示灰度图像

	if (src.empty())
	{
		cout << "invalid image path" << endl;
		return;
	}

	cv::namedWindow("input", cv::WINDOW_FREERATIO); //WINDOW_FREERATIO表示大小可调，cv::WINDOW_AUTOSIZE为默认，原图大小保持一致
	cv::imshow("input",src);
	cv::waitKey(0);  //0表示一直停止，等待按键，数字为等待毫秒数
	cv::destroyAllWindows();
	return;
}

//颜色空间转换
void colorSpaceDemo(cv::Mat &img)
{
	cv::Mat gray, hsv;
	cv::cvtColor(img, hsv, COLOR_BGR2HSV);  //H 0 - 180 S V 0 -255
	cv::cvtColor(img, gray, COLOR_BGR2GRAY);
	cv::imshow("HSV", hsv);
	cv::imshow("gray", gray);
	cv::imwrite("./img/hsv.jpg", hsv);
	cv::imwrite("./img/gray.jpg", gray);

	cv::waitKey(0);

	return;
}

//创建Mat对象
void matCreation(cv::Mat &img)
{
	cv::Mat m1, m2;
	m1 = img.clone();
	img.copyTo(img);

	//创建空白图像
	cv::Mat m3 = cv::Mat::zeros(cv::Size(8, 8), CV_8UC3);
	m3 = cv::Scalar(100,100,100);
	cout << "Width: " << m3.cols << endl;
	cout << "Height: " << m3.rows << endl;
	cout << "channels: " << m3.channels() << endl;

	return;
}

//单个像素访问
void pixelVisitDemo(cv::Mat &img)
{
	int w = img.cols;
	int h = img.rows;
	int dims = img.channels();
	for(int row=0; row<h; row++)
		for (int col = 0; col < w; col++)
		{
			if (dims == 1)//灰度
			{
				int pv = img.at<uchar>(row, col);   //at访问
				img.at<uchar>(row, col) = 255 - pv;
			}
			else if (dims == 3)//彩色
			{
				cv::Vec3b bgr = img.at<cv::Vec3b>(row, col);
				img.at<cv::Vec3b>(row, col)[0] = 255 - bgr[0];
				img.at<cv::Vec3b>(row, col)[1] = 255 - bgr[1];
				img.at<cv::Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}

	//指针方式访问
	for (int row = 0; row < h; row++)
	{
		uchar* currentRow = img.ptr<uchar>(row);
		for (int col = 0; col < w; col++)
		{
			if (dims == 1)
			{
				int pv = *currentRow;
				*currentRow++ = 255 - pv;
			}
			else if (dims == 3)
			{
				*currentRow++ = 255 - *currentRow;
				*currentRow++ = 255 - *currentRow;
				*currentRow++ = 255 - *currentRow;
			}
		}

	}

	cv::imshow("pixel read&write",img);
	cv::waitKey(0);
}


//图像像素操作（加减乘除）
void operatorDemo(cv::Mat &img)
{
	cv::Mat dst;
	dst = img + cv::Scalar(10, 10, 10); //add  乘法不支持

	cv::Mat m = cv::Mat::zeros(img.size(),img.type());
	m = cv::Scalar(2, 2, 2);
	cv::multiply(img, m, dst);  //专用乘法  add subtract divide

	//cv::saturate_cast<uchar>()  把数值范围限定在0-255之间


	cv::imshow("add", dst);
	cv::waitKey(0);
}

// trackbar 滚动条操作
#if 1

static void on_track(int b, void* userdata)
{
	cv::Mat image = *((cv::Mat *)userdata);
	cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
	cv::Mat m = cv::Mat::zeros(image.size(), image.type());
	m = cv::Scalar(b, b, b);
	cv::add(image, m, dst);
	cv::imshow("lightness adjust", dst);
}

void tracking_bar_demo(cv::Mat &image)
{
	namedWindow("lightness adjust", cv::WINDOW_AUTOSIZE);
	int max_value = 100;
	int lightness = 50;
	cv::createTrackbar("Value Bar:", "lightness adjust", &lightness, max_value, on_track, (void *)(&image));
	on_track(50, &image);
}

#endif


/*waitkey解析
1.waitKey()与waitKey(0)，都代表无限等待，waitKey函数的默认参数就是int delay = 0，故这俩形式本质是一样的。
2.waitKey(n)，等待n毫秒后，关闭显示的窗口。
3.当等待时间内无任何操作时等待结束后返回-1。
4.当等待时间内有输入字符时，则返回输入字符的ASCII码对应的十进制值。
*/
void key_demo(cv::Mat &image)
{
	cv:Mat dst = cv::Mat::zeros(image.size(), image.type());
	image.copyTo(dst);
	while (true)
	{
		int c = cv::waitKey(100);

		std::cout << c << std::endl;

		if (c == 27)
			break;
		else if (c == '1')
			cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
		else if (c == '2')
			image.copyTo(dst);

		cv::imshow("waitKey test", dst);

	}
}

//图像像素位操作
void bitwise_demo(cv::Mat &image)
{
	cv::Mat m1 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
	cv::Mat m2 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);

	cv::rectangle(m1, cv::Rect(100, 100, 80, 80), cv::Scalar(255, 255, 0), -1);
	cv::rectangle(m2, cv::Rect(150, 150, 80, 80), cv::Scalar(0, 255, 255), -1);

	cv::imshow("m1", m1);
	cv::imshow("m2", m2);

	cv::Mat dst;
	bitwise_and(m1, m2, dst);
	bitwise_not(image, dst);
	cv::imshow("bit operate", dst);
}

//通道操作
void channels_demo(cv::Mat &image)
{
	std::vector<cv::Mat> mv;
	cv::split(image, mv);
	imshow("blue", mv[0]);
	imshow("green", mv[1]);
	imshow("red", mv[2]);

	cv::Mat dst;
	mv[1] = 0;
	mv[2] = 0;
	cv::merge(mv, dst);
	imshow("blue", dst);

	int from_to[] = { 0,2,1,1,2,0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("mix", dst);
}


//色彩空间转换
// H: 0-180  S: 0-255 V: 0-255
void inrange_demo(cv::Mat &image)
{
	cv::Mat hsv;
	cv::cvtColor(image, hsv, COLOR_BGR2HSV);
	cv::Mat mask;
	cv::inRange(hsv, cv::Scalar(35, 43, 46), cv::Scalar(77, 255, 255), mask);  //绿色,可查hsv表获取颜色范围

	cv::Mat redback = cv::Mat(image.size(), image.type());
	redback = cv::Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	cv::imshow("mask", mask);
	image.copyTo(redback, mask);   //只对redback的mask（白色像素点）区域有作用
}


//像素统计
void pixel_statistic_demo(cv::Mat &image)
{
	double minv, maxv;
	cv::Point minLoc, maxLoc;
	std::vector<cv::Mat> mv;
	cv::split(image, mv); //minMaxLoc必须用在单通道上
	cv::minMaxLoc(mv[0], &minv, &maxv, &minLoc, &maxLoc, cv::Mat());

	std::cout << "min value: " << minv << " max value: " << maxv << std::endl;

	cv::Mat mean, stddev;
	cv::meanStdDev(image, mean, stddev);
	//mean.at<double>(1,0);
	std::cout << "mean: " << mean << " stddev: " << stddev << std::endl;
}

//图形绘制
void draw_demo(cv::Mat &image)
{
	cv::Rect rect;
	rect.x = 200;
	rect.y = 200;
	rect.width = 100;
	rect.height = 100;
	cv::Mat bg = cv::Mat::zeros(image.size(), image.type());
	cv::rectangle(bg, rect, cv::Scalar(0, 0, 255), 2, 8, 0);
	cv::circle(bg, cv::Point(350, 400), 15, cv::Scalar(255, 0, 0), -1, 8, 0);

	cv::line(bg, cv::Point(100, 100), cv::Point(350, 400), 2, 8,0);

	cv::Mat dst;

	cv::addWeighted(image, 0.5, bg, 0.4, 0, dst);
	cv::imshow("draw", dst);
}

void polyline_drawing_demo()
{
	cv::Mat canvas =  cv::Mat::zeros(cv::Size(512, 512), CV_8UC3);
	cv::Point p1(100, 100);
	cv::Point p2(350, 100);
	cv::Point p3(450, 280);
	cv::Point p4(320, 450);
	cv::Point p5(80, 400);

	std::vector<cv::Point> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
	//cv::fillPoly(canvas, pts, cv::Scalar(255, 255, 0), 8, 0);
	cv::polylines(canvas, pts, true, cv::Scalar(0, 0, 255), 2, LINE_AA, 0);
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(pts);
	cv::drawContours(canvas, contours,-1,cv::Scalar(255,0,255),-1);  //填充

	cv::imshow("poly drawing", canvas);

}

#if 1  //鼠标事件
cv::Point sp(-1, -1);
cv::Point ep(-1, -1);

cv::Mat temp;

static void on_draw(int event, int x, int y, int flags, void *userdata)
{
	cv::Mat image = *((cv::Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN)
	{
		sp.x = x;
		sp.y = y;
		std::cout << "start point: " << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0 && x <= image.cols && y <= image.rows)
		{
			cv::Rect box(sp.x, sp.y, dx, dy);
			temp.copyTo(image);
			imshow("ROI", image(box));
			cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::imshow("mouse drawing", image);
			//ready for nex drawing
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (sp.x > 0 && sp.y > 0)
		{
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0 && x <= image.cols-1 && x <= image.rows)
			{
				cv::Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8, 0);
				cv::imshow("mouse drawing", image);
			}
		}
	}
}

void mouse_drawing_demo(cv::Mat &image)
{
	cv::namedWindow("mouse drawing", WINDOW_AUTOSIZE);
	cv::setMouseCallback("mouse drawing", on_draw,(void*)(&image));
	cv::imshow("mouse drawing", image);

	temp = image.clone();
}

#endif

//归一化
void norm_demo(cv::Mat &image)
{
	cv::Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);
	std::cout << dst.type() << std::endl;
	cv::normalize(image, dst, 1.0, 0, NORM_MINMAX);

	cv::imshow("normalize", dst);
}

//图像缩放
void resize_demo(cv::Mat &image)
{
	cv::Mat zoomIn, zoomOut;
	int h = image.rows;
	int w = image.cols;
	cv::resize(image, zoomOut, cv::Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	cv::imshow("zoomOut", zoomOut);
}

//图像翻转
void flip_demo(cv::Mat &image)
{
	cv::Mat dst;
	cv::flip(image, dst, 0); //上下翻转
	//cv::flip(image, dst, 1); //左右翻转
	//cv::flip(image, dst, -1); //180度旋转
	cv::imshow("image flip", dst);
}

//图像旋转
void rotate_demo(cv::Mat &image)
{
	cv::Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	M = cv::getRotationMatrix2D(cv::Point2f(w / 2, h / 2), 45, 1.0);  //旋转中心 ， 角度， 放缩比例
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);  //原图无损失旋转
	//warpAffine(image, dst, M, image.size());
	warpAffine(image, dst, M, cv::Size(nw, nh), INTER_LINEAR, 0, cv::Scalar(255, 255, 0));
	cv::imshow("rotate", dst);
}

//摄像头
void video_demo()
{
	//cv::VideoCapture capture(0); //摄像头
	cv::VideoCapture capture("E:/movie/dji/DJI_0287.MP4"); //文件
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH); //视频宽度
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT); //视频高度
	int count = capture.get(CAP_PROP_FRAME_COUNT); //总帧数
	int fps = capture.get(CAP_PROP_FPS);

	cv::VideoWriter writer("./test.mp4", capture.get(CAP_PROP_FOURCC), fps, cv::Size(frame_width, frame_height), true);

	cv::Mat frame;
	while (true)
	{
		capture.read(frame);  //之后可对视频进行一些处理
		cv::flip(frame, frame, 1);
		if (frame.empty())
		{
			break;
		}
		cv::namedWindow("frame", cv::WINDOW_FREERATIO); //WINDOW_FREERATIO表示大小可调，cv::WINDOW_AUTOSIZE为默认，原图大小保持一致
		cv::imshow("frame", frame);

		writer.write(frame);

		int c = cv::waitKey(1);
		if (c == 27)
			break;
	}
	capture.release();
	writer.release();
}

void histogram_eq_demo(cv::Mat &image)
{
	cv::Mat gray;
	cv::cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat dst;
	cv::equalizeHist(gray, dst);
	cv::imshow("hist", dst);
}
void blur_demo(cv::Mat &image)
{
	cv::Mat dst;
	cv::blur(image, dst, cv::Size(3, 3), cv::Point(-1, -1));
	cv::imshow("blur", dst);
}

void gaussian_blur_demo(cv::Mat &image)
{
	cv::Mat dst;
	cv::GaussianBlur(image, dst, cv::Size(3, 3), 15);
	cv::imshow("gaussian", dst);
}

//双边高斯模糊,美颜效果
void bifilter_demo(cv::Mat &image)
{
	cv::Mat dst;
	cv::bilateralFilter(image, dst, 0, 100, 10);
	imshow("bilateral", dst);
}

//顶帽操作

void tophat_test(void)
{
	cv::Mat img = cv::imread("F:/c++/test_opencv/test_opencv/test_opencv/img/credit.png");
	if (img.empty())
		return;
	cv::Mat gray;
	cv::cvtColor(img, gray, COLOR_BGR2GRAY);

	cv::Mat result;
	cv::threshold(gray, result, 170, 255, cv::THRESH_BINARY_INV);
	cv::threshold(gray, result, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

	//获取自定义核
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的

	cv::Mat erodeImg,dilateImg;
	cv::erode(gray, erodeImg, element);
	cv::dilate(erodeImg, dilateImg, element); //相当于一次开操作

	cv::Mat topHatImg;

	topHatImg = gray - dilateImg;  //相当于一次顶帽操作

	return;

}

//泊松融合
#include <opencv2/photo.hpp>
void possionMerge(void)
{
	Mat src = imread("images/iloveyouticket.jpg");
	Mat dst = imread("images/wood-texture.jpg");

	// Create an all white mask
	Mat src_mask = 255 * Mat::ones(src.rows, src.cols, src.depth());

	// The location of the center of the src in the dst
	Point center(dst.cols / 2, dst.rows / 2);

	// Seamlessly clone src into dst and put the results in output
	Mat normal_clone;
	Mat mixed_clone;


	seamlessClone(src, dst, src_mask, center, normal_clone, NORMAL_CLONE);
	seamlessClone(src, dst, src_mask, center, mixed_clone, MIXED_CLONE);
}

#include <opencv2/reg/mappergradaffine.hpp> 
#include <opencv2/reg/mapperpyramid.hpp> 
//配准并返回与模板对比后的二值图，适合重叠度较高的两张图片配准
void affineRegCompare1(const cv::Mat& img1_8u, const cv::Mat& img2_8u, std::string curname)
{

	cv::Mat img1, img2;
	cv::Mat img1_8u_c1, img2_8u_c1;


	//cv::cvtColor(img1_8u, img1_8u_c1, CV_BGR2GRAY);
	//cv::cvtColor(img2_8u, img2_8u_c1, CV_BGR2GRAY);
	//img1_8u_c1.convertTo(img1, CV_64FC1);
	//img2_8u_c1.convertTo(img2, CV_64FC1);

	std::string productID = "1";// m_modelname;    //机种号

	img1_8u.convertTo(img1, CV_64FC1);
	img2_8u.convertTo(img2, CV_64FC1);

	cv::Mat dest;

	// Register
	cv::Ptr<cv::reg::MapperGradAffine> mapper = cv::makePtr<cv::reg::MapperGradAffine>();
	cv::reg::MapperPyramid mappPyr(mapper);
	cv::Ptr<cv::reg::Map> mapPtr = mappPyr.calculate(img1, img2);

	// Display registration accuracy
	mapPtr->inverseWarp(img2, dest);				//模板与原图配准

	//dest.convertTo(dest, CV_8UC3);
	//img1.convertTo(img1, CV_8UC3);

	img1.convertTo(img1, CV_8UC1);
	dest.convertTo(dest, CV_8UC1);

	cv::imwrite("./template/" + productID + "/template_" + curname + "_temp.bmp", dest);

	return;

}

int main(void)
{
	//Mat img = cv::imread("F:/c++/test_opencv/test_opencv/img/ayukawa.jpg"); //IMREAD_GRAYSCALE

	Mat img = cv::imread("F:/python_project/Deep-Learning-Approach-for-Surface-Defect-Detection-master/visualization/training_epoch-153/kos30_Part3.jpg"); //IMREAD_GRAYSCALE

	string path = "F:/c++/test_opencv/test_opencv/img/ayukawa.jpg";

	//showImg(path);

	//colorSpaceDemo(img);

	//matCreation(img);
	//pixelVisitDemo(img);

	//operatorDemo(img);
	//tracking_bar_demo(img);

	//key_demo(img);

	//bitwise_demo(img);

	//channels_demo(img);

	//inrange_demo(img);

	//pixel_statistic_demo(img);

	//draw_demo(img);

	//polyline_drawing_demo();

	//mouse_drawing_demo(img);

	//norm_demo(img);

	//resize_demo(img);

	//rotate_demo(img);

	//video_demo();
	//histogram_eq_demo(img);

	//blur_demo(img);

	//bifilter_demo(img);


	tophat_test();

	cv::waitKey(0);
	
	return 0;
}
