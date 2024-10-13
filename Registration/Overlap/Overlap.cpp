#include<pcl/registration/correspondence_estimation.h> 
//点云重合率计算  para1 = 0.5   para2 =  0.15  

#include <pcl/point_types.h>  
#include <pcl/io/pcd_io.h>  
#include <pcl/registration/correspondence_estimation.h>  
#include <iostream>  
#include <cmath>  
 
using namespace pcl;  
using namespace std;  
 
class IterationMatch {  
public:  
    static void calaPointCloudCoincide(PointCloud<PointXYZ>::Ptr cloud_src, PointCloud<PointXYZ>::Ptr cloud_target, float para1, float para2, float &coincide);  
};  


void IterationMatch::calaPointCloudCoincide(PointCloud::Ptr cloud_src, PointCloud::Ptr cloud_target,  float para1, float para2, float &coincide)
{
	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> core;
	core.setInputSource(cloud_src);
	core.setInputTarget(cloud_target);

	boost::shared_ptr<pcl::Correspondences> cor(new pcl::Correspondences);   //共享所有权的智能指针，以kdtree做索引

	core.determineReciprocalCorrespondences(*cor, para1);   //点之间的最大距离,cor对应索引

	//构造重叠点云的PCD格式文件
	PointCloud overlapA;
	PointCloud overlapB;

	overlapA.width = cor->size();
	overlapA.height = 1;
	overlapA.is_dense = false;
	overlapA.points.resize(overlapA.width*overlapA.height);

	overlapB.width = cor->size();
	overlapB.height = 1;
	overlapB.is_dense = false;
	overlapB.points.resize(overlapB.width * overlapB.height);
	cout << "点云原来的数量：" << cloud_target->size() << endl;
	cout << "重合的点云数： " << cor->size() << endl;
	double num = 0;
	for (size_t i = 0; i < cor->size(); i++)
	{
		//overlapA写入pcd文件
		overlapA.points[i].x = cloud_src->points[cor->at(i).index_query].x;
		overlapA.points[i].y = cloud_src->points[cor->at(i).index_query].y;
		overlapA.points[i].z = cloud_src->points[cor->at(i).index_query].z;

		//overlapB写入pcd文件
		overlapB.points[i].x = cloud_target->points[cor->at(i).index_match].x;
		overlapB.points[i].y = cloud_target->points[cor->at(i).index_match].y;
		overlapB.points[i].z = cloud_target->points[cor->at(i).index_match].z;
		
		double dis = sqrt(pow(overlapA.points[i].x - overlapB.points[i].x, 2) + 
			pow(overlapA.points[i].y - overlapB.points[i].y, 2) +
			pow(overlapA.points[i].z - overlapB.points[i].z, 2));
		if (dis < para2)
			num++;
	}


	cout << "精配重叠区域的点云数：" << num << endl;
	cout << "重合率： " << float(num / cor->size())*100<< "%"<< endl;
	coincide = float(num / cor->size());

}


int main(int argc, char** argv) {  
    PointCloud<PointXYZ>::Ptr cloud_src(new PointCloud<PointXYZ>);  
    PointCloud<PointXYZ>::Ptr cloud_target(new PointCloud<PointXYZ>);  
 
    // 加载点云数据，这里应该使用你的点云文件路径  
    if (io::loadPCDFile<PointXYZ>("source.pcd", *cloud_src) == -1 ||  
        io::loadPCDFile<PointXYZ>("target.pcd", *cloud_target) == -1) {  
        PCL_ERROR("Couldn't read file source.pcd or target.pcd \n");  
        return (-1);  
    }  
 
    float coincide = 0.0f;  
    IterationMatch::calaPointCloudCoincide(cloud_src, cloud_target, 0.5f, 0.15f, coincide);  
 
    cout << "Final coincidence rate: " << coincide * 100 << "%" << endl;  
 
    return 0;  
}