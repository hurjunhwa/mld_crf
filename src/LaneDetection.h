struct LANE_MARKING {
	cv::Point2f str_p;
	cv::Point2f cnt_p;
	cv::Point2f end_p;
	cv::Point2f inn_p;
	int size;
};
struct MARKING_SEED {
	std::vector<int> index;
	int flag;	// 1: valid results, 0: candidnates, -1: invalid supermarking
	float cnt_dir;
	float str_dir;
	float end_dir;
	float length;
	cv::Point2f cnt_p;
	cv::Point2f str_p;
	cv::Point2f end_p;
};
struct NODE_CRF {
	int vert_idx1;
	int vert_idx2;
	int label;	// 
	int idx;	// group #
	double unary;
	//int edge;//??
};
struct EDGE_CRF {
	std::vector<int> node_idx;	// nodes index
	int grp_idx;		// group # that it belongs to
	int label;
	float pairwise;
};
struct NODE_GRP {
	std::vector<int> idx;	// node #
};

class LaneDetection {

public:

	LaneDetection() {
	}
	~LaneDetection() {
	}

	bool initialize_variable(std::string& img_name);
	bool initialize_Img(std::string& img_name);
	void lane_marking_detection(bool verbose = false);
	float marking_thres(float input);
	int dist_ftn1(int i, int sj, double slope);
	
	void seed_generation(bool verbose = false);
	void seed_specification(MARKING_SEED& marking_seed_curr, int mode);
	float dist_ftn2(int idx1, int idx2);
	float slope_ftn(cv::Point2f i, cv::Point2f j);
	float length_ftn(cv::Point2f str_p, cv::Point2f end_p);
	
	void graph_generation(bool verbose);
	int dist_ftn3(int i, int j, int s_i, int s_j);
	float unary_ftn(int i, int j);
	void node_grouping(cv::Mat& mat_in, int size, int type, int n, int label);
	float pairwise_ftn(std::vector<cv::Point2f>& pts);
	void validating_final_seeds(bool verbose);

	float poly4(std::vector<cv::Point2f> points, int n, std::vector<float>& coeff);
	float poly3(std::vector<cv::Point2f> points, int n, std::vector<float>& coeff);
	float poly2(std::vector<cv::Point2f> points, int n, std::vector<float>& coeff);

	//void display_test1(IplImage*);
	//void display_test2(IplImage*);
	//void memory_release();

private:

	// Image
	cv::Size img_size;
	cv::Mat img_gray;
	int img_height;
	int img_width;
	int img_roi_height;
	int img_depth;

	// Lane marking variable
	std::vector<int> max_lw;
	std::vector<int> min_lw;
	std::vector<int> max_lw_d;
	std::vector<LANE_MARKING> lm;
	std::vector<MARKING_SEED> marking_seed;
	// Marking detection
	//int* ELW;
	//int num_of_ELW;

	// Graphical Model
	std::vector<NODE_CRF> nodes;
	std::vector<EDGE_CRF> edges;
};