#pragma warning(disable: 4819)
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "LaneDetection.h"


// Lane marking definition 
#define MAX_LANE_MARKING 2000
#define MAX_LW_N 40		// Max lane width, nearest
#define MAX_LW_F 8		// Max lane width, farest
#define MAX_LW_D 10		// Max lane width, delta
#define MIN_LW_N 20 		// Min lane width, nearest
#define MIN_LW_F 2			// Min lane width, farest
#define SCAN_INTERVAL1 1	//lower
#define SCAN_INTERVAL2 1	//upper

// Lane Marking Grouping
#define MAX_LANE_SEED 200
#define SEED_MARKING_DIST_THRES 90
#define VALID_SEED_MARKING_NUMBER_THRES 6
#define LOW_LEVEL_ASS_THRES 1.96

// Initializing variables depending on the resolution of the input image.

double valueAt(std::vector<float>& f, float x) {
	float ans = 0.f;
	for (int i = (int)f.size() - 1; i >= 0; --i)
		ans = ans * x + f[i];
	return ans;
}

bool LaneDetection::initialize_variable(std::string& img_name) {

	// Image variable setting
	cv::Mat img_src = cv::imread(img_name);
	if (img_src.empty()) {
		std::cout << "Err: Cannot find an input image for initialization: " << img_name << std::endl;
		return false;
	}

	img_size = img_src.size();
	img_height = img_src.rows;
	img_width = img_src.cols;
	img_roi_height = (int)(img_size.height*0.5);
	img_depth = img_src.depth();

	max_lw.resize(img_height);
	min_lw.resize(img_height);
	max_lw_d.resize(img_width);

	// Estimated Lane Width
	for (int hh = img_roi_height; hh < img_height; ++hh) {
		max_lw[hh] = (int)((MAX_LW_N - MAX_LW_F)*(hh - img_roi_height) / (img_size.height - img_roi_height) + MAX_LW_F);
		min_lw[hh] = (int)((MIN_LW_N - MIN_LW_F)*(hh - img_roi_height) / (img_size.height - img_roi_height) + MIN_LW_F);
	}

	int w = img_width - 1;
	while (img_width - 1 - w < w) {
		max_lw_d[w] = (int)(MAX_LW_D*(abs(w - (img_width - 1) / 2.0)) / ((img_width - 1) / 2.0));
		max_lw_d[img_width - 1 - w] = (int)(MAX_LW_D*(abs(w - (img_width - 1) / 2.0)) / ((img_width - 1) / 2.0));
		w--;
	}

	return true;
}

bool LaneDetection::initialize_Img(std::string& img_name) {

	// Loading an input image
	cv::Mat img_src = cv::imread(img_name);
	if (img_src.empty()) {
		std::cout << "Err: Cannot find the input image: " << img_name << std::endl;
		return false;
	}

	img_gray = cv::Mat(img_size, img_depth);
	if (img_src.channels() == 1) {
		img_src.copyTo(img_gray);
	}
	else {
		cv::cvtColor(img_src, img_gray, CV_BGR2GRAY);
	}

	// Variable initialization 
	lm.resize(0);
	marking_seed.resize(0);
	nodes.resize(0);
	edges.resize(0);
	return true;
}

void LaneDetection::lane_marking_detection(bool verbose) {


	for (int h = img_roi_height; h < img_height;) {

		// half size of the filter
		int hf_size = 2 + 8 * (h - img_roi_height + 1) / (img_height - img_roi_height);

		std::vector<int> scan_line(img_width);

		// Edge Extraction
		for (int w = hf_size + 1; w < img_width - hf_size - 1; w++) {

			// left edge value, right edge value
			int l_val = 0;
			int r_val = 0;

			for (int i = -hf_size; i<0; i++) {
				l_val = l_val + img_gray.at<uchar>(h, w + i);
			}
			for (int i = 1; i <= hf_size; i++) {
				r_val = r_val + img_gray.at<uchar>(h, w + i);
			}
			if (((float)(r_val - l_val) / (float)hf_size)>marking_thres((float)l_val / (float)hf_size)) scan_line[w] = 1; // left edge = 1;
			if (((float)(l_val - r_val) / (float)hf_size)>marking_thres((float)r_val / (float)hf_size)) scan_line[w] = -1; // right edge = -1;
		}

		// Edge Centering
		int e_flag = 0; // edge flag
		for (int w = hf_size + 1; w < img_width - hf_size - 2; w++) {
			if (scan_line[w] == 1) {
				if (e_flag >= 0) {
					e_flag++;
				}
				else {
					scan_line[w - (int)(e_flag / 2.0)] = -10;
					e_flag = 0;
				}
			}
			else if (scan_line[w] == -1) {
				if (e_flag <= 0) {
					e_flag--;
				}
				else {
					scan_line[w + (int)(e_flag / 2.0)] = 10;
					e_flag = 0;
				}
			}
			else {
				if (e_flag > 0) {
					scan_line[w - (int)(e_flag / 2.0)] = 10;
					e_flag = 0;
				}
				else if (e_flag < 0) {
					scan_line[w + (int)(e_flag / 2.0)] = -10;
					e_flag = 0;
				}
			}
		}

		// Extracting Lane Markings - marking flag
		cv::Point2i l_pt, r_pt;
		int m_flag = 0;

		for (int w = hf_size + 1; w < img_width - hf_size - 1; w++) {
			if (scan_line[w] == 10) {
				m_flag = 1;
				l_pt.x = w;
				l_pt.y = h;
			}
			if (m_flag == 1) {
				if (scan_line[w] == -10) {
					m_flag = 2;
					r_pt.x = w;
					r_pt.y = h;
				}
			}
			if (m_flag == 2) {
				if (((r_pt.x - l_pt.x) >= min_lw[h]) && ((r_pt.x - l_pt.x) <= (max_lw[h] + max_lw_d[w]))) {

					// lane update
					LANE_MARKING lm_new;
					lm_new.str_p = l_pt;
					lm_new.end_p = r_pt;
					lm_new.cnt_p.x = (int)((l_pt.x + r_pt.x) / 2.0);
					lm_new.cnt_p.y = r_pt.y;
					if (lm_new.cnt_p.x > (int)(img_size.width / 2)) {
						lm_new.inn_p = l_pt;
					}
					else {
						lm_new.inn_p = r_pt;
					}
					lm_new.size = r_pt.x - l_pt.x;
					lm.push_back(lm_new);
					w = r_pt.x + 5;
					m_flag = 0;
					if (lm.size() >= MAX_LANE_MARKING - 1) {
						break;
					}
				}
				m_flag = 0;
			}
		}
		if (lm.size() >= MAX_LANE_MARKING - 1) {
			break;
		}

		//if (h < 120) {
		//	h += SCAN_INTERVAL1;
		//}
		//else {
		//	h += SCAN_INTERVAL2;
		//}
		h += SCAN_INTERVAL1;
	}

	if (verbose) {
		cv::Mat img_test = cv::Mat(img_size, CV_8UC3);
		for (int n = 0; n < lm.size(); n++) {
			cv::line(img_test, lm[n].str_p, lm[n].end_p, CV_RGB(0, 255, 0), 2, 8, 0);
		}
		cv::imshow("Lane marking detection", img_test);
	}
}

void LaneDetection::seed_generation(bool verbose) {

	// Initialization

	// STEP 1-1. Generating Seeds: Making a bunch of seeds consisting of lane markings near each others.
	int flag_group = 0;
	int flag_dist = 0;
	for (int ii = 0; ii < lm.size(); ii++) {
		flag_group = 0;
		for (int jj = marking_seed.size() - 1; jj >= 0; jj--) {

			flag_dist = dist_ftn1(ii, marking_seed[jj].index[marking_seed[jj].index.size() - 1], marking_seed[jj].cnt_dir);

			if (flag_dist == 1) {
				flag_group = 1;
				marking_seed[jj].index.push_back(ii);
				if (marking_seed[jj].cnt_dir < -99) {
					marking_seed[jj].cnt_dir = slope_ftn(lm[ii].cnt_p, marking_seed[jj].cnt_p);
				}
				else {
					marking_seed[jj].cnt_dir = 0.8*marking_seed[jj].cnt_dir + 0.2*slope_ftn(lm[ii].cnt_p, marking_seed[jj].cnt_p);
				}
				marking_seed[jj].cnt_p = lm[ii].cnt_p;

				break;
			}
		}
		if (flag_group == 0) {
			MARKING_SEED seed_new;
			seed_new.flag = 0;
			seed_new.index.resize(0);
			seed_new.index.push_back(ii);
			seed_new.cnt_dir = -100;
			seed_new.cnt_p = lm[ii].cnt_p;
			marking_seed.push_back(seed_new);
		}
	}

	if (verbose) {
		cv::Mat img_test_marking_seed = cv::Mat(img_size, CV_8UC3);
		for (int ii = 0; ii < marking_seed.size(); ++ii) {
			int	r = rand() % 200 + 50;
			int	g = rand() % 200 + 50;
			int b = rand() % 200 + 50;
			for (int jj = 0; jj < marking_seed[ii].index.size(); ++jj) {
				int idx = marking_seed[ii].index[jj];
				cv::line(img_test_marking_seed, lm[idx].str_p, lm[idx].end_p, CV_RGB(r, g, b), 2, 8, 0);
			}	
		}
		cv::imshow("Raw marking seeds", img_test_marking_seed);
	}

	// STEP 1-2. Seed Validation
	int count_i, count_j;
	float var;
	for (int ii = 0; ii < marking_seed.size(); ii++) {
		count_i = marking_seed[ii].index.size();

		// if contained lane marking is less then a certain number
		if (count_i < VALID_SEED_MARKING_NUMBER_THRES) {
			marking_seed[ii].flag = -1;
			continue;
		}
		if (count_i < 10) {
			float mean = 0.f;
			for (int jj = 0; jj < count_i; jj++) {
				int idx_i = marking_seed[ii].index[jj];
				mean = mean + lm[idx_i].size;
			}
			mean = (float)mean / (float)count_i;
			float var = 0.f;
			for (int jj = 0; jj < count_i; jj++) {
				int idx_i = marking_seed[ii].index[jj];
				var = var + (lm[idx_i].size - mean)*(lm[idx_i].size - mean);
			}
			var = var / (float)count_i;

			// if variance is higher, it regarded as invalid
			if (var > 6.0) {
				marking_seed[ii].flag = -1;
			}
		}
	}

	// STEP 1-3. Seed specification: Getting information of each seeds, position & direction
	std::vector<int> val_seed;

	srand((unsigned)time(NULL));
	int r, g, b;
	for (int ii = 0; ii < marking_seed.size(); ii++) {
		if (marking_seed[ii].flag < 0) {
			continue;
		}
		seed_specification(marking_seed[ii], 1);
		val_seed.push_back(ii);
	}

	if (verbose) {
		cv::Mat img_test_valid_seed = cv::Mat(img_size, CV_8UC3);
		for (int ii = 0; ii < val_seed.size(); ++ii) {
			int	r = rand() % 200 + 50;
			int	g = rand() % 200 + 50;
			int b = rand() % 200 + 50;

			MARKING_SEED seed = marking_seed[val_seed[ii]];
			for (int jj = 0; jj < seed.index.size(); ++jj) {
				int idx = seed.index[jj];
				cv::line(img_test_valid_seed, lm[idx].str_p, lm[idx].end_p, CV_RGB(r, g, b), 2, 8, 0);
			}
			//std::cout << " [" << ii << "]" << std::endl;
			//std::cout << marking_seed[val_seed[ii]].str_p << "  " << marking_seed[val_seed[ii]].cnt_p << "  " << marking_seed[val_seed[ii]].end_p << std::endl;
			//std::cout << marking_seed[val_seed[ii]].str_dir << "  " << marking_seed[val_seed[ii]].cnt_dir << "  " << marking_seed[val_seed[ii]].end_dir << " " << marking_seed[val_seed[ii]].index.size() << std::endl;
			//cv::imshow("val marking seeds", img_test_valid_seed);
			//cv::waitKey(0);
		}
		cv::imshow("val marking seeds", img_test_valid_seed);
	}

	// STEP 2. Seed Growing - Dist_mat Generation
	int n_of_valid_seeds = val_seed.size();
	std::vector<int> trns_stats;
	trns_stats.resize(n_of_valid_seeds, -1);
	cv::Mat dist_mat = cv::Mat(n_of_valid_seeds, n_of_valid_seeds, CV_32FC1);

	for (int ii = 0; ii < n_of_valid_seeds; ++ii) {
		dist_mat.at<float>(ii, ii) = -1.f;
		for (int jj = ii + 1; jj < n_of_valid_seeds; ++jj) {
			dist_mat.at<float>(ii, jj) = dist_ftn2(val_seed[ii], val_seed[jj]);
			dist_mat.at<float>(jj, ii) = dist_mat.at<float>(ii, jj);
		}
	}

	//for (int ii = 0; ii < n_of_valid_seeds; ++ii) {
	//	for (int jj = 0; jj < n_of_valid_seeds; ++jj) {
	//		std::cout << dist_mat.at<float>(ii, jj) << " ";
	//	}
	//	std::cout << std::endl;
	//}



	// STEP 2-1. Low Level Association Process #1 - Head -> Tail
	for (int ii = 0; ii < n_of_valid_seeds; ++ii) {
		int cnct_count = 0;
		int cnct_idx = -1;
		for (int jj = 0; jj < ii; ++jj) {
			if (dist_mat.at<float>(jj, ii) > LOW_LEVEL_ASS_THRES) {
				cnct_count++;
				cnct_idx = jj;
			}
		}
		int valid_flag = 0;
		float temp_max = 0;
		int max_id = -1;

		if (cnct_count == 1) {
			for (int kk = cnct_idx; kk<n_of_valid_seeds; kk++) {
				if (dist_mat.at<float>(cnct_idx, kk) > temp_max) {
					temp_max = dist_mat.at<float>(cnct_idx, kk);
					max_id = kk;
				}
			}
			if (max_id == ii) {
				valid_flag = 1;
			}
		}
		if (valid_flag == 1) {
			//	The only seed which comes down to 'cnct_idx' is 'ii'. Thus, 'cnct_idx' has to be connected to 'ii'.
			MARKING_SEED* seed_dst = &marking_seed[val_seed[ii]];
			MARKING_SEED* seed_connect = &marking_seed[val_seed[cnct_idx]];
			count_j = seed_connect->index.size();
			for (int kk = 0; kk < count_j; kk++) {
				seed_dst->index.push_back(seed_connect->index[kk]);
			}
			seed_connect->index.resize(0);
			seed_dst->flag = 1;
			seed_connect->flag = -1;	// seed # which become included in i
			seed_specification(*seed_dst, 0);
			seed_dst->str_dir = seed_connect->str_dir;
			seed_dst->str_p = seed_connect->str_p;
			seed_dst->length = seed_dst->length + seed_connect->length;
			for (int ll = cnct_idx; ll < n_of_valid_seeds; ll++) {
				dist_mat.at<float>(cnct_idx, ll) = 0;
			}
			// remember where the transition happened
			trns_stats[cnct_idx] = ii;
		}
	}

	int temp_val = 0;
	int last_idx = 0;
	// STEP 2-2. Low Level Association Process #2 - Head <- Tail
	for (int ii = n_of_valid_seeds - 1; ii >= 0; ii--) {
		int cnct_count = 0;
		int cnct_idx = -1;
		for (int jj = ii + 1; jj < n_of_valid_seeds; jj++) {
			if (dist_mat.at<float>(ii, jj) > LOW_LEVEL_ASS_THRES) {
				cnct_count++;
				cnct_idx = jj;
			}
		}
		int valid_flag = 0;
		int temp_max = 0;
		int max_id = -1;
		if (cnct_count == 1) {
			for (int kk = 0; kk<cnct_idx; kk++) {
				if (dist_mat.at<float>(kk, cnct_idx) > temp_max) {
					temp_max = dist_mat.at<float>(kk, cnct_idx);
					max_id = kk;
				}
			}
			if (max_id == ii) {
				valid_flag = 1;
			}
		}
		if (valid_flag == 1) {
			// remember where the transition happened
			last_idx = cnct_idx;
			temp_val = trns_stats[last_idx];
			while (temp_val != -1) {
				last_idx = temp_val;
				temp_val = trns_stats[last_idx];
			}
			cnct_idx = last_idx;
			// the only seed coming upto 'cnct_idx' is 'i'.
			MARKING_SEED* seed_dst = &marking_seed[val_seed[ii]];
			MARKING_SEED* seed_connect = &marking_seed[val_seed[cnct_idx]];
			count_j = seed_connect->index.size();
			for (int kk = 0; kk < count_j; kk++) {
				seed_dst->index.push_back(seed_connect->index[kk]);
			}
			seed_connect->index.resize(0);
			seed_dst->flag = 1;
			seed_connect->flag = -1;
			seed_specification(*seed_dst, 0);
			seed_dst->end_dir = seed_connect->end_dir;
			seed_dst->end_p = seed_connect->end_p;
			seed_dst->length = seed_dst->length + seed_connect->length;
			for (int ll = 0; ll < cnct_idx; ll++) {
				dist_mat.at<float>(ll, cnct_idx) = 0;
			}
		}
	}

	if (verbose) {
		cv::Mat img_test_raw_level_assoc = cv::Mat(img_size, CV_8UC3);
		for (int ii = 0; ii < marking_seed.size(); ++ii) {
			if (marking_seed[ii].flag < 0) {
				continue;
			}
			int	r = rand() % 200 + 50;
			int	g = rand() % 200 + 50;
			int b = rand() % 200 + 50;

			MARKING_SEED seed = marking_seed[ii];
			for (int jj = 0; jj < seed.index.size(); ++jj) {
				int idx = seed.index[jj];
				cv::line(img_test_raw_level_assoc, lm[idx].str_p, lm[idx].end_p, CV_RGB(r, g, b), 2, 8, 0);
			}
		}
		cv::imshow("Low Level Association", img_test_raw_level_assoc);
	}
}

void LaneDetection::seed_specification(MARKING_SEED& marking_seed_curr, int mode) {

	float temp_x = 0;
	float temp_y = 0;

	std::vector<float> coeff2;
	std::vector<cv::Point2f> points;
	coeff2.resize(2);
	int n_of_lm = marking_seed_curr.index.size();

	for (int ii = 0; ii < n_of_lm; ii++) {
		int idx_lm = marking_seed_curr.index[ii];
		temp_x += (float)lm[idx_lm].cnt_p.x;
		temp_y += (float)lm[idx_lm].cnt_p.y;
		points.push_back(lm[idx_lm].cnt_p);
	}
	poly2(points, points.size(), coeff2);
	marking_seed_curr.cnt_dir = CV_PI / 2 - atan(coeff2[1]);
	marking_seed_curr.cnt_p.x = (int)(temp_x / n_of_lm);
	marking_seed_curr.cnt_p.y = (int)(temp_y / n_of_lm);

	if (mode == 1) {	// initial seed
		marking_seed_curr.str_p = lm[marking_seed_curr.index[0]].cnt_p;
		marking_seed_curr.end_p = lm[marking_seed_curr.index[n_of_lm - 1]].cnt_p;
		marking_seed_curr.length = length_ftn(marking_seed_curr.str_p, marking_seed_curr.end_p);
		if (n_of_lm < VALID_SEED_MARKING_NUMBER_THRES) {
			marking_seed_curr.end_dir = marking_seed_curr.cnt_dir;
			marking_seed_curr.str_dir = marking_seed_curr.cnt_dir;
		}
		else {
			int n_samples = std::max(5, (int)(0.3f*n_of_lm));
			poly2(points, n_samples, coeff2);
			marking_seed_curr.str_dir = (float)(CV_PI / 2 - atan(coeff2[1]));
			points.resize(0);
			for (int ii = n_of_lm - 1; ii >= n_of_lm - n_samples; ii--) {
				int idx_i = marking_seed_curr.index[ii];
				points.push_back(lm[idx_i].cnt_p);
			}
			poly2(points, n_samples, coeff2);
			marking_seed_curr.end_dir = (float)(CV_PI / 2 - atan(coeff2[1]));
		}
	}

	//printf("%d %d / %d %d\n", marking_seed[idx].str_p.x, marking_seed[idx].str_p.y, marking_seed[idx].end_p.x, marking_seed[idx].end_p.y);
	// the lowest point(in human frame) is the start point, vice versa

}

void LaneDetection::graph_generation(bool verbose) {

	srand((unsigned)time(NULL));

	cv::Mat img_test_graph = cv::Mat(img_size, CV_8UC3);

	// STEP 1. Graph Formulation
	std::vector<int> grp_seed;

	for (int ii = 0; ii < marking_seed.size(); ii++) {
		if (marking_seed[ii].flag < 0) {
			continue;
		}
		if (marking_seed[ii].index.size() < VALID_SEED_MARKING_NUMBER_THRES) {
			continue;
		}
		grp_seed.push_back(ii);
	}


	// STEP 2-1. Node Generation - Generating valid node 
	int n_of_grp_seeds = grp_seed.size();
	cv::Mat vert_mat = cv::Mat(n_of_grp_seeds, n_of_grp_seeds, CV_32SC1);
	std::vector<int> row_sum(n_of_grp_seeds);
	std::vector<int> col_sum(n_of_grp_seeds);
	std::vector<int> ele_sum(n_of_grp_seeds);

	for (int ii = 0; ii < n_of_grp_seeds; ii++) {
		for (int jj = 0; jj < n_of_grp_seeds; jj++) {
			vert_mat.at<int>(ii, jj) = dist_ftn3(grp_seed[ii], grp_seed[jj], ii, jj);
		}
		vert_mat.at<int>(ii, ii) = -1;
	}

	//for (int hh = 0; hh < n_of_grp_seeds; ++hh) {
	//	for (int ww = 0; ww < n_of_grp_seeds; ++ww) {
	//		std::cout << vert_mat.at<int>(hh, ww) << " ";
	//	}
	//	std::cout << std::endl;
	//}

	// STEP 2-2. Separating nodes to each groups
	int n_of_node_grps = 0;
	for (int ii = 0; ii < n_of_grp_seeds; ii++) {
		for (int jj = 0; jj < n_of_grp_seeds; jj++) {
			if (vert_mat.at<int>(ii, jj) == 1) {
				vert_mat.at<int>(ii, jj) = n_of_node_grps + 100;
				node_grouping(vert_mat, n_of_grp_seeds, 0, ii, n_of_node_grps + 100);
				node_grouping(vert_mat, n_of_grp_seeds, 0, jj, n_of_node_grps + 100);
				node_grouping(vert_mat, n_of_grp_seeds, 1, jj, n_of_node_grps + 100);
				node_grouping(vert_mat, n_of_grp_seeds, 1, ii, n_of_node_grps + 100);
				n_of_node_grps++;
			}
		}
	}

	//for (int hh = 0; hh < n_of_grp_seeds; ++hh) {
	//	for (int ww = 0; ww < n_of_grp_seeds; ++ww) {
	//		std::cout << vert_mat.at<int>(hh, ww) << " ";
	//	}
	//	std::cout << std::endl;
	//}

	// STEP 2-3. Node indexing & initialization
	nodes.resize(0);
	for (int ii = 0; ii < n_of_grp_seeds; ii++) {
		for (int jj = 0; jj < n_of_grp_seeds; jj++) {
			if (vert_mat.at<int>(ii, jj) >= 100) {
				NODE_CRF node_new;
				node_new.vert_idx1 = ii;
				node_new.vert_idx2 = jj;
				node_new.idx = vert_mat.at<int>(ii, jj) - 100;
				
				// Node initialization - Unary Term
				node_new.unary = unary_ftn(grp_seed[ii], grp_seed[jj]);
				nodes.push_back(node_new);
			}
		}
	}
	
	// STEP 2-4. Node Grouping
	std::vector<NODE_GRP> node_grp(n_of_node_grps);
	for (int ii = 0; ii < nodes.size(); ii++) {
		int node_grp_idx = nodes[ii].idx;
		node_grp[node_grp_idx].idx.push_back(ii);
	}
	
	//// Grouping result display
	//for (int i = 0; i < n_of_node_grps; i++) {
	//	for (int j = 0; j < node_grp[i].idx.size(); j++) {
	//		printf("%d ", node_grp[i].idx[j]);
	//	}
	//	printf("\n");
	//}

	// Hungarian Method
		// 1) Sorting! in the order of Unary term - Unary term:Logistic function, Sorting - bubble sort
		// 2) Labling using the Constraint - with clear rules! with 4)
		// 3) Calculating the pairwise term with finding Edges - Nodes which are in the same group have the same edges
		// 4) iteration back to the 1) - clear rules, with 2)


	// STEP 3. Hungarian Methos, Edge Indexing, Initialization	

	for (int nn = 0; nn < n_of_node_grps; nn++) {
		// STEP 3-1. Sorting! in the order of Unary term - Unary term:Logistic function, Sorting - bubble sort
		for (int ii = node_grp[nn].idx.size() - 1; ii > 0; ii--) {
			for (int jj = 0; jj < ii; jj++) {
				if (nodes[node_grp[nn].idx[jj]].unary < nodes[node_grp[nn].idx[jj + 1]].unary) {
					int temp_val = node_grp[nn].idx[jj + 1];
					node_grp[nn].idx[jj + 1] = node_grp[nn].idx[jj];
					node_grp[nn].idx[jj] = temp_val;
				}
			}
		}
		//// debugging
		//printf(" > Sorting node grp [%d]:", nn);
		//for (int i = 0; i < node_grp[nn].idx.size(); i++) {
		//	printf(" %d", node_grp[nn].idx[i]);
		//}
		//printf("\n");

		if (node_grp[nn].idx.size() == 1) {	// trivial case which doesn't need the inference 
			continue;
		}
		for (int n_iter = 0; n_iter < node_grp[nn].idx.size(); n_iter++) {
			// STEP 3-2. For each iteration in Hungarian Methods, Find the possible edges
			for (int ii = 0; ii < n_of_grp_seeds; ii++) {
				row_sum[ii] = 0;
				col_sum[ii] = 0;
				ele_sum[ii] = 0;
			}
			nodes[node_grp[nn].idx[n_iter]].label = 1;
			int n_of_labels = 1;
			row_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx1]++;
			col_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx2]++;
			ele_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx1]++;
			ele_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx2]++;

			for (int ii = 0; ii < node_grp[nn].idx.size(); ++ii) {
				if (ii == n_iter) {
					continue;
				}
				if (row_sum[nodes[node_grp[nn].idx[ii]].vert_idx1]>0) {
					nodes[node_grp[nn].idx[ii]].label = 0;
					continue;
				}
				if (col_sum[nodes[node_grp[nn].idx[ii]].vert_idx2]>0) {
					nodes[node_grp[nn].idx[ii]].label = 0;
					continue;
				}
				nodes[node_grp[nn].idx[ii]].label = 1;
				n_of_labels++;
				row_sum[nodes[node_grp[nn].idx[ii]].vert_idx1]++;
				col_sum[nodes[node_grp[nn].idx[ii]].vert_idx2]++;
				ele_sum[nodes[node_grp[nn].idx[ii]].vert_idx1]++;
				ele_sum[nodes[node_grp[nn].idx[ii]].vert_idx2]++;
			}
			// Discarding those which cannot construct an edge
			if (n_of_labels <= 1) {
				continue;
			}
			// Indexing nodes consisting of the edge
			EDGE_CRF edge_new;
			for (int ii = 0; ii < node_grp[nn].idx.size(); ++ii) {
				if (nodes[node_grp[nn].idx[ii]].label == 1) {
					edge_new.node_idx.push_back(node_grp[nn].idx[ii]);
				}
			}
			int iden_flag = 0;	// 0 if different, 1 if identical
			for (int ii = 0; ii < edges.size(); ii++) {
				if (edges[ii].node_idx.size() != edge_new.node_idx.size()) {
					//printf("  >> disciarding <-> %d - count\n", i);
					iden_flag = 0;
					continue;
				}
				for (int jj = 0; jj < edge_new.node_idx.size(); jj++) {
					if (edges[ii].node_idx[jj] != edge_new.node_idx[jj]) {
						//printf("  >> disciarding <-> %d - index %d\n", i, j);
						iden_flag = 0;
						break;
					}
					iden_flag = 1;
				}
				if (iden_flag == 1) {
					break;
				}
			}

			if ((edges.size() != 0) && (iden_flag == 1)) {
				continue;	// this edges is already included
			}

			// STEP 3-3. Pairwise cost calculation
			int n_of_pts = 0;
			std::vector<cv::Point2f> pts;
			for (int ii = 0; ii < n_of_grp_seeds; ii++) {
				if (ele_sum[ii] > 0) {
					int count_i = marking_seed[grp_seed[ii]].index.size();
					for (int jj = 0; jj < count_i; jj++) {
						if (count_i > 15) {
							if (jj % (count_i / 14) != 0) {
								continue;
							}
						}
						cv::Point2f pts_new;
						pts_new.x = (float)lm[marking_seed[grp_seed[ii]].index[jj]].cnt_p.x;
						pts_new.y = (float)lm[marking_seed[grp_seed[ii]].index[jj]].cnt_p.y;
						pts.push_back(pts_new);
					}
				}
			}

			edge_new.pairwise = pairwise_ftn(pts);
			edge_new.grp_idx = nn;
			edges.push_back(edge_new);

			//std::cout << "  > pairwise [" << edges.size() << "] : " << std::endl;
			//for (int tt = 0; tt < edges[edges.size() - 1].node_idx.size(); tt++) {
			//	std::cout << "    " << edges[edges.size() - 1].node_idx[tt] << " " << edges[edges.size() - 1].pairwise << " " << edges[edges.size() - 1].grp_idx << std::endl;
			//}
			//std::cout << std::endl;
		}
	}

	// CRF Formulation, Hungarian Method	
	//printf("\n ==== Hungarian Method ==== \n");
	std::vector<int> final_label;
	final_label.resize(nodes.size(), -1);

	double energy = 0;
	double min_energy = 0;
	int expt_flag = 0;
	for (int nn = 0; nn < n_of_node_grps; nn++) {

		min_energy = 0;
		//printf(" > grp #%d\n\n", nn);
		for (int n_iter = 0; n_iter<node_grp[nn].idx.size(); n_iter++) {
			//printf(" >> iter #%d\n", n_iter);
			// Exception # 1
			if (node_grp[nn].idx.size() == 1) {
				if (nodes[node_grp[nn].idx[0]].unary > 0.5) {
					final_label[node_grp[nn].idx[0]] = 1;
				}
				else {
					final_label[node_grp[nn].idx[0]] = 0;
				}
				continue;
			}
			// Exception # 2
			expt_flag = 0;
			for (int ii = 0; ii<node_grp[nn].idx.size(); ii++) {
				if (nodes[node_grp[nn].idx[ii]].unary > 0.5) {
					break;
				}
				if (ii == node_grp[nn].idx.size() - 1) {
					expt_flag = 1;
				}
			}
			if (expt_flag == 1) {
				continue;
			}

			// 2) Labling using the Constraint - with clear rules! with 4)
			for (int ii = 0; ii < n_of_grp_seeds; ii++) {
				row_sum[ii] = 0;
				col_sum[ii] = 0;
				ele_sum[ii] = 0;
			}
			nodes[node_grp[nn].idx[n_iter]].label = 1;
			int n_of_labels = 1;
			row_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx1]++;
			col_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx2]++;
			ele_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx1]++;
			ele_sum[nodes[node_grp[nn].idx[n_iter]].vert_idx2]++;
			for (int ii = 0; ii < node_grp[nn].idx.size(); ++ii) {
				if (ii == n_iter) {
					continue;
				}
				if (row_sum[nodes[node_grp[nn].idx[ii]].vert_idx1]>0) {
					nodes[node_grp[nn].idx[ii]].label = 0;
					continue;
				}
				if (col_sum[nodes[node_grp[nn].idx[ii]].vert_idx2]>0) {
					nodes[node_grp[nn].idx[ii]].label = 0;
					continue;
				}
				nodes[node_grp[nn].idx[ii]].label = 1;
				n_of_labels++;
				row_sum[nodes[node_grp[nn].idx[ii]].vert_idx1]++;
				col_sum[nodes[node_grp[nn].idx[ii]].vert_idx2]++;
				ele_sum[nodes[node_grp[nn].idx[ii]].vert_idx1]++;
				ele_sum[nodes[node_grp[nn].idx[ii]].vert_idx2]++;
			}

			//printf("  >> # of labels: %d\n", n_of_labels);
			//printf("  >> nodes: ");
			//for(int ii=0;ii<node_grp[nn].idx.size();ii++){
			//	printf("[%d] %d / ", node_grp[nn].idx[ii], nodes[node_grp[nn].idx[ii]].label);
			//}
			//printf("\n");

			// 3) Calculating the pairwise term after finding Edges - Nodes which are in the same group have the common edges
			//printf("  >> edges: ");
			for (int ii = 0; ii < edges.size(); ii++) {
				if (edges[ii].grp_idx != nn) {
					continue;
				}
				if (edges[ii].node_idx.size() != n_of_labels) {
					edges[ii].label = 0;
					continue;
				}
				for (int jj = 0; jj < edges[ii].node_idx.size(); jj++) {
					if (nodes[edges[ii].node_idx[jj]].label != 1) {
						edges[ii].label = 0;
						break;
					}
					edges[ii].label = 1;
				}
				//printf("[%d] %d / ",ii,edges[ii].label);
			}
			//printf("\n");

			////  Calculating Energy & Updating labels
			energy = 0;
			for (int ii = 0; ii < node_grp[nn].idx.size(); ++ii) {
				if (nodes[node_grp[nn].idx[ii]].label == 1) {
					energy = energy - nodes[node_grp[nn].idx[ii]].unary;
				}
				else {
					energy = energy - (1 - nodes[node_grp[nn].idx[ii]].unary);
				}
			}
			expt_flag = 0;
			for (int ii = 0; ii < edges.size(); ++ii) {
				if (edges[ii].grp_idx != nn) {
					continue;
				}
				if (edges[ii].label == 1) {
					energy = energy - edges[ii].pairwise;
					if (edges[ii].pairwise < 0.5) {
						expt_flag = 1;
					}
				}
				else {
					energy = energy - (1 - edges[ii].pairwise);
				}
			}
			//printf(" >>> energy = %.3f, min = %.3f\n", energy, min_energy);				

			if (energy < min_energy) {
				min_energy = energy;
				for (int ii = 0; ii < node_grp[nn].idx.size(); ++ii) {
					final_label[node_grp[nn].idx[ii]] = nodes[node_grp[nn].idx[ii]].label;
					if (expt_flag == 1) {
						final_label[node_grp[nn].idx[ii]] = 0;
					}
					//printf(" >>> [%d] %d\n", node_grp[nn].idx[ii], nodes[node_grp[nn].idx[ii]].label);
				}
			}
			//printf("\n");
			//cvShowImage("Pairwise", testImg);
			//cvWaitKey(0);
		}         
	}

	//// Final Result Printing
	//printf(" >> Labels : ");
	//for (int ii = 0; ii < nodes.size(); ++ii) {
	//	nodes[ii].label = final_label[ii];
	//	printf("%d ", nodes[ii].label);
	//}
	//printf("\n");

	//// Graph Print
	////printf(" > after graph\n");
	////for(int i=0;i<n_of_grp_seeds;i++){
	////	for(int j=0;j<n_of_grp_seeds;j++){
	////		printf("%d ", vert_mat[i][j]);
	////	}
	////	printf("\n");
	////}
	////

	// Seeds association according to the nodes those having been labeled by 1 : s_i -> s_j ( s_i is absorbed into s_j )
	int s_i, s_j;
	int count_j;
	for (int ii = 0; ii < nodes.size(); ++ii) {
		if (nodes[ii].label != 1) {
			continue;
		}
		s_i = grp_seed[nodes[ii].vert_idx1];
		s_j = grp_seed[nodes[ii].vert_idx2];
		count_j = marking_seed[s_j].index.size();
		for (int jj = 0; jj < marking_seed[s_i].index.size(); ++jj) {
			marking_seed[s_j].index.push_back(marking_seed[s_i].index[jj]);
		}

		marking_seed[s_j].flag = 1;
		marking_seed[s_i].flag = -1;
		seed_specification(marking_seed[s_j], 0);
		marking_seed[s_j].str_dir = marking_seed[s_i].str_dir;
		marking_seed[s_j].str_p = marking_seed[s_i].str_p;
		marking_seed[s_j].length = marking_seed[s_i].length + marking_seed[s_j].length;
	}

	// Test displaying
	if (verbose) {
		cv::Mat img_test_crf = cv::Mat(img_size, CV_8UC3);
		for (int ii = 0; ii < marking_seed.size(); ++ii) {
			if (marking_seed[ii].flag < 0) {
				continue;
			}
			int r = rand() % 230 + 20;
			int g = rand() % 230 + 20;
			int b = rand() % 230 + 20;

			for (int jj = 0; jj < marking_seed[ii].index.size(); ++jj) {
				int temp_i = marking_seed[ii].index[jj];
				cv::line(img_test_crf, lm[temp_i].str_p, lm[temp_i].end_p, cv::Scalar(b, g, r), 2, 8, 0);
			}
		}
		cv::imshow("CRF", img_test_crf);
	}
}

void LaneDetection::validating_final_seeds(bool verbose) {

	cv::Mat img_test_val = cv::Mat(img_size, CV_8UC3);
	
	for (int ii = 0; ii < marking_seed.size(); ii++) {
		if ((marking_seed[ii].flag == 0) && (marking_seed[ii].index.size()>23)) {
			marking_seed[ii].flag = 1;
		}
		if (marking_seed[ii].flag < 1) {
			continue;
		}
		float length = length_ftn(marking_seed[ii].end_p, marking_seed[ii].str_p);
		if (length < 50) {
			marking_seed[ii].flag = 0;
			continue;
		}
		if (marking_seed[ii].length < 50) {
			marking_seed[ii].flag = 0;
			continue;
		}
		if (marking_seed[ii].length / length < 0.382) {
			marking_seed[ii].flag = 0;
			continue;
		}
		if ((length == marking_seed[ii].length) && (length < 62)) {
			marking_seed[ii].flag = 0;
			continue;
		}

		// supermarking displaying
		int r = rand() % 230 + 20;
		int g = rand() % 230 + 20;
		int b = rand() % 230 + 20;

		for (int jj = 0; jj < marking_seed[ii].index.size(); ++jj) {
			int temp_i = marking_seed[ii].index[jj];
			cv::line(img_test_val, lm[temp_i].str_p, lm[temp_i].end_p, cv::Scalar(b, g, r), 2, 8, 0);
		}


		// polynomial fitting
		std::vector<cv::Point2f> pts;
		std::vector<float> coeff(3);
		cv::Point2f dot_p;
		for (int pp = 0; pp < marking_seed[ii].index.size(); pp++) {
			int idx_lm = marking_seed[ii].index[pp];
			pts.push_back(lm[idx_lm].cnt_p);
		}
		poly3(pts, pts.size(), coeff);
		
		for (int yy = marking_seed[ii].str_p.y; yy < marking_seed[ii].end_p.y; ++yy) {
			dot_p.y = yy;
			dot_p.x = valueAt(coeff, dot_p.y);
			cv::circle(img_test_val, dot_p, 1, cv::Scalar(0, 255, 0), 1, 8, 0);
		}

		cv::imshow("final", img_test_val);
	}
	

	

}

float LaneDetection::marking_thres(float input) {

	float thres = 0;

	/*if(input<50){
	thres = (int)(input/10);
	}else{
	thres = (int)(15+input/200*10);
	}*/
	//return thres;

	return input / 10 + 4;
}
int LaneDetection::dist_ftn1(int s_i, int s_j, double slope) {

	// For Seed Generation

	double value = 0;
	double slope_new = slope_ftn(lm[s_i].cnt_p, lm[s_j].cnt_p);
	CvPoint i, j;
	i = lm[s_i].cnt_p;
	j = lm[s_j].cnt_p;
	value = (i.x - j.x)*(i.x - j.x) + (i.y - j.y)*(i.y - j.y);


	if ((lm[s_i].str_p.x > lm[s_j].end_p.x) || (lm[s_i].end_p.x < lm[s_j].str_p.x)) {
		//printf(">> location err (%d,%d) (%d,%d) \n", lm[s_i].str_p.x, lm[s_i].end_p.x,lm[s_j].str_p.x, lm[s_j].end_p.x);
		return 0;
	}

	//printf("  >> slope : %.3f, diff : %.3f, location : (%d,%d) (%d,%d)", slope, abs(slope-slope_new),  lm[s_i].str_p.x, lm[s_i].end_p.x,lm[s_j].str_p.x, lm[s_j].end_p.x);

	if (value < SEED_MARKING_DIST_THRES) {
		if (slope <= -99) {
			//printf(">> initial\n");
			return 1;
		}
		if ((value>50) && (abs(slope - slope_new) > 1.1)) {
			return 0;
		}
		if (abs(slope - slope_new) < 0.8) {
			//printf(">> slope %.3f\n",abs(slope-slope_new));
			return 1;
		}
		if ((lm[s_i].cnt_p.x <= lm[s_j].end_p.x) && (lm[s_i].cnt_p.x >= lm[s_j].str_p.x)) {
			//printf(">> location\n");
			return 1;
		}
	}
	return 0;
}
float LaneDetection::dist_ftn2(int i, int j) {

	// For Low level Association
	if (marking_seed[i].end_p.y > marking_seed[j].str_p.y) {
		return 0;
	}

	// Rough Verification
	std::vector<float> slp;
	slp.resize(7);
	slp[0] = marking_seed[i].cnt_dir;
	slp[1] = marking_seed[j].cnt_dir;
	if ((abs(slp[0] - slp[1])>0.5) && (abs(abs(slp[0] - slp[1]) - 3.141592) < 2.641592)) {
		return 0;
	}
	slp[2] = slope_ftn(marking_seed[i].cnt_p, marking_seed[j].cnt_p);
	slp[3] = slope_ftn(marking_seed[i].str_p, marking_seed[j].str_p);
	slp[4] = slope_ftn(marking_seed[i].str_p, marking_seed[j].end_p);
	slp[5] = slope_ftn(marking_seed[i].end_p, marking_seed[j].str_p);
	slp[6] = slope_ftn(marking_seed[i].end_p, marking_seed[j].end_p);

	// slope variance check
	float slp_mean = (slp[0] + slp[1] + slp[2] + slp[3] + slp[4] + slp[5] + slp[6]) / 7;
	float temp = 0;
	for (int i = 0; i < 7; i++) {
		temp += (slp[i] - slp_mean)*(slp[i] - slp_mean);
	}
	float slp_var = temp / 7;
	if (slp_var > 0.5) {
		return 0;
	}

	// distance ftn between two seeds	
	float sig = 0.25;
	float diff1, diff2;
	diff1 = slp[0] - slp[2];
	diff2 = slp[1] - slp[2];
	// it should be that 1 < 3 < 2 or 2 < 3 < 1
	if (((abs(diff1) + abs(diff2)) > 0.2) && (diff1*diff2 > 0)) {
		return 0;
	}
	if (abs(diff1) > 1.570796) {
		diff1 = abs(diff1 - 3.141592);
	}
	if (abs(diff2) > 1.570796) {
		diff2 = abs(diff2 - 3.141592);
	}

	return (float)(exp(-(diff1)*(diff1) / sig*sig) + exp(-(diff2)*(diff2) / sig*sig));
}


int LaneDetection::dist_ftn3(int i, int j, int s_i, int s_j) {

	// Graph Validity of (i to j)

	// Location 1
	if (marking_seed[i].end_p.y >= marking_seed[j].str_p.y) {
		return 0;
	}

	//printf(" >> Node [%d] -> [%d]\n",s_i,s_j);

	// Location 2
	double diff1 = marking_seed[j].str_p.x - (tan(CV_PI / 2 - marking_seed[i].end_dir)*(marking_seed[j].str_p.y - marking_seed[i].end_p.y) + marking_seed[i].end_p.x);
	double diff2 = marking_seed[i].end_p.x - (tan(CV_PI / 2 - marking_seed[j].str_dir)*(marking_seed[i].end_p.y - marking_seed[j].str_p.y) + marking_seed[j].str_p.x);
	//printf("  >> location diff = \t%.3f\t%.3f\t  %.3f\n", abs(diff1), abs(diff2), abs(diff1)+abs(diff2));
	if (abs(diff1) + abs(diff2) > 65) {
		//printf("  >> location diff [%d] -> [%d] = %.3f\t%.3f\t  %.3f\n", s_i,s_j, abs(diff1), abs(diff2), abs(diff1)+abs(diff2));
		return 0;
	}

	// Slope
	double inter_dir = slope_ftn(marking_seed[i].end_p, marking_seed[j].str_p);
	double diff3 = (marking_seed[i].end_dir - inter_dir) / CV_PI * 180;
	double diff4 = (marking_seed[j].str_dir - inter_dir) / CV_PI * 180;
	//printf("  >> slope diff = \t%.3f\t%.3f\t%.3f\t\t%.3f\t%.3f\t%.3f\n", abs(diff3), abs(diff4), abs(diff3)+abs(diff4), inter_dir/CV_PI*180, marking_seed[i].end_dir/CV_PI*180, marking_seed[j].str_dir/CV_PI*180);
	if (abs(diff3) + abs(diff4) > 80) {
		//printf("  >> slope diff [%d] -> [%d] = %.3f\t%.3f\t  %.3f\t\t%.3f\t%.3f\t%.3f\n", s_i,s_j, abs(diff3), abs(diff4), abs(diff3)+abs(diff4), inter_dir/CV_PI*180, marking_seed[i].end_dir/CV_PI*180, marking_seed[j].str_dir/CV_PI*180);
		return 0;
	}

	//printf(" >> [%d] -> [%d] \n", s_i,s_j);
	return 1;

	// possible to be resued for the Unary term
}

float LaneDetection::slope_ftn(cv::Point2f pos1, cv::Point2f pos2) {

	cv::Point2f temp_pos;
	if (pos1.y > pos2.y) {
		temp_pos = pos1;
		pos1 = pos2;
		pos2 = temp_pos;
	}
	return (float)(acos((double)((pos2.x - pos1.x) / sqrt((float)((pos1.x - pos2.x)*(pos1.x - pos2.x) + (pos1.y - pos2.y)*(pos1.y - pos2.y))))));
}
float LaneDetection::length_ftn(cv::Point2f str_p, cv::Point2f end_p) {

	return sqrt((float)(str_p.x - end_p.x)*(str_p.x - end_p.x) + (float)(str_p.y - end_p.y)*(str_p.y - end_p.y));

}

float LaneDetection::unary_ftn(int i, int j) {

	// Location diff
	double diff1 = marking_seed[j].str_p.x - (tan(CV_PI / 2 - marking_seed[i].end_dir)*(marking_seed[j].str_p.y - marking_seed[i].end_p.y) + marking_seed[i].end_p.x);
	double diff2 = marking_seed[i].end_p.x - (tan(CV_PI / 2 - marking_seed[j].str_dir)*(marking_seed[i].end_p.y - marking_seed[j].str_p.y) + marking_seed[j].str_p.x);

	// Slope diff
	double inter_dir = slope_ftn(marking_seed[i].end_p, marking_seed[j].str_p);
	double diff3 = (marking_seed[i].end_dir - inter_dir) / CV_PI * 180;
	double diff4 = (marking_seed[j].str_dir - inter_dir) / CV_PI * 180;

	double x = abs(diff1) + abs(diff2);
	double y = abs(diff3) + abs(diff4);
	double unary = 0;
	//printf(" %.3f %.3f %.3f\n", term1,term2, unary);

	double a, b, c, d, e, f;
	a = 0.000140047;
	b = 0.001069285;
	c = -0.000263005;
	d = -0.283444141;
	e = -0.255786389;
	f = 24.86101278;

	double fx = a*x*x + b*x*y + c*y*y + d*x + e*y + f;
	unary = 1 / (1 + exp(-fx));
	return (float)unary;
}

float LaneDetection::pairwise_ftn(std::vector<cv::Point2f>& pts) {

	cv::Point2f dots; 
	std::vector<float> coeff(4);
	float error = 0;
	poly4(pts,pts.size(),coeff);
	for(int ii=0;ii<pts.size();++ii){
		dots.y = (int)pts[ii].y;
		dots.x = (int)(coeff[0]+coeff[1]*dots.y+coeff[2]*dots.y*dots.y+coeff[3]*dots.y*dots.y*dots.y);
		error = error + (float)((pts[ii].x-dots.x)*(pts[ii].x-dots.x));
	}
	
	double sig = 50;
	double pairwise = exp(-(error/ pts.size())*(error/ pts.size())/sig/sig);
	
	return (float)pairwise;

}
void LaneDetection::node_grouping(cv::Mat& mat_in, int size, int type, int n, int label) {

	if (type == 0) {
		for (int ii = 0; ii < size; ii++) {
			if (mat_in.at<int>(n, ii) == 1) {
				mat_in.at<int>(n, ii) = label;
				node_grouping(mat_in, size, 0, ii, label);
				node_grouping(mat_in, size, 1, ii, label);
			}
		}
	}

	if (type == 1) {
		for (int ii = 0; ii < size; ii++) {
			if (mat_in.at<int>(ii, n) == 1) {
				mat_in.at<int>(ii, n) = label;
				node_grouping(mat_in, size, 0, ii, label);
				node_grouping(mat_in, size, 1, ii, label);
			}
		}
	}
}


float LaneDetection::poly2(std::vector<cv::Point2f> points, int n, std::vector<float>& coeff) {

	float norm_f = 1.f;
	float temp;
	double err;
	cv::Mat a = cv::Mat(2, 2, CV_32FC1);
	cv::Mat b = cv::Mat(2, 1, CV_32FC1);
	cv::Mat c = cv::Mat(2, 2, CV_32FC1);
	cv::Mat d = cv::Mat(2, 1, CV_32FC1);
	cv::Mat e = cv::Mat(2, 1, CV_32FC1);

	for (int ii = 0; ii < n; ii++) {
		points[ii].x = points[ii].x / norm_f;
		points[ii].y = points[ii].y / norm_f;
	}

	// configuring matrix 'a'
	a.at<float>(0, 0) = (float)n;
	temp = 0;
	for (int ii = 0; ii < n; ii++) {
		temp += points[ii].y;
	}
	a.at<float>(0, 1) = (float)temp;
	a.at<float>(1, 0) = (float)temp;
	temp = 0;
	for (int ii = 0; ii < n; ii++) {
		temp += points[ii].y * points[ii].y;
	}
	a.at<float>(1, 1) = (float)temp;
	//temp = 0;
	//for (int ii = 0; ii < n; ii++) {
	//	temp += points[ii].y * points[ii].y * points[ii].y;
	//}

	// configuring matrix 'b'
	temp = 0;
	for (int ii = 0; ii < n; ii++) {
		temp += points[ii].x;
	}
	b.at<float>(0, 0) = (float)temp;
	temp = 0;
	for (int ii = 0; ii < n; ii++) {
		temp += points[ii].x * points[ii].y;
	}
	b.at<float>(1, 0) = (float)temp;

	// matrix operation
	c = a.inv();
	d = c*b;
	coeff[0] = d.at<float>(0, 0)*norm_f;
	coeff[1] = d.at<float>(1, 0)*norm_f;

	//printf("%f %f %f\n", coeff[0], coeff[1], coeff[2]); 

	e = a*d;
	err = abs(e.at<float>(0, 0) - b.at<float>(0, 0)) + abs(e.at<float>(1, 0) - b.at<float>(1, 0));

	return err;
}
float LaneDetection::poly3(std::vector<cv::Point2f> points, int n, std::vector<float>& coeff) {

	float norm_f = 1.f;
	float temp;
	float err = 0;
	cv::Mat a = cv::Mat(3, 3, CV_32FC1);
	cv::Mat b = cv::Mat(3, 1, CV_32FC1);
	cv::Mat c = cv::Mat(3, 3, CV_32FC1);
	cv::Mat d = cv::Mat(3, 1, CV_32FC1);
	cv::Mat e = cv::Mat(3, 1, CV_32FC1);

	for (int ii = 0; ii < n; ii++) {
		points[ii].x = points[ii].x / norm_f;
		points[ii].y = points[ii].y / norm_f;
	}
	// configuring matrix 'a'
	a.at<float>(0, 0) = (float)n;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y;
	}
	a.at<float>(0, 1) = (float)temp;
	a.at<float>(1, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y;
	}
	a.at<float>(0, 2) = (float)temp;
	a.at<float>(1, 1) = (float)temp;
	a.at<float>(2, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y;
	}
	a.at<float>(1, 2) = (float)temp;
	a.at<float>(2, 1) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y * points[i].y;
	}
	a.at<float>(2, 2) = (float)temp;

	// configuring matrix 'b'
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].x;
	}
	b.at<float>(0, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].x * points[i].y;
	}
	b.at<float>(1, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].x;
	}
	b.at<float>(2, 0) = (float)temp;

	// matrix operation
	c = a.inv();
	d = c*b;
	coeff[0] = d.at<float>(0, 0)*norm_f;
	coeff[1] = d.at<float>(1, 0)*norm_f;
	coeff[2] = d.at<float>(2, 0) / norm_f;

	e = a*d;
	err = abs(e.at<float>(0, 0) - b.at<float>(0, 0)) + abs(e.at<float>(1, 0) - b.at<float>(1, 0)) + abs(e.at<float>(2, 0) - b.at<float>(2, 0));

	return err;
}

float LaneDetection::poly4(std::vector<cv::Point2f> points, int n, std::vector<float>& coeff) {

	float norm_f = (float)20.f;
	float temp;
	double err = 0;
	cv::Mat a = cv::Mat(4, 4, CV_32FC1);
	cv::Mat b = cv::Mat(4, 1, CV_32FC1);
	cv::Mat c = cv::Mat(4, 4, CV_32FC1);
	cv::Mat d = cv::Mat(4, 1, CV_32FC1);
	cv::Mat e = cv::Mat(4, 1, CV_32FC1);

	for (int i = 0; i < n; i++) {
		points[i].x = points[i].x / norm_f;
		points[i].y = points[i].y / norm_f;
	}
	// configuring matrix 'a'
	a.at<float>(0, 0) = (float)n;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y;
	}
	a.at<float>(0, 1) = (float)temp;
	a.at<float>(1, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y;
	}
	a.at<float>(0, 2) = (float)temp;
	a.at<float>(1, 1) = (float)temp;
	a.at<float>(2, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y;
	}
	a.at<float>(0, 3) = (float)temp;
	a.at<float>(1, 2) = (float)temp;
	a.at<float>(2, 1) = (float)temp;
	a.at<float>(3, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y * points[i].y;
	}
	a.at<float>(1, 3) = (float)temp;
	a.at<float>(2, 2) = (float)temp;
	a.at<float>(3, 1) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y * points[i].y * points[i].y;
	}
	a.at<float>(2, 3) = (float)temp;
	a.at<float>(3, 2) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y * points[i].y * points[i].y * points[i].y;
	}
	a.at<float>(3, 3) = (float)temp;

	// configuring matrix 'b'
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].x;
	}
	b.at<float>(0, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].x * points[i].y;
	}
	b.at<float>(1, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].x;
	}
	b.at<float>(2, 0) = (float)temp;
	temp = 0;
	for (int i = 0; i < n; i++) {
		temp += points[i].y * points[i].y * points[i].y * points[i].x;
	}
	b.at<float>(3, 0) = (float)temp;

	// matrix operation
	c = a.inv();
	d = c*b;
	//printf("\n>> %f %f %f ", cvmGet(d,0,0),cvmGet(d,1,0),cvmGet(d,2,0));
	coeff[0] = d.at<float>(0, 0) * norm_f;
	coeff[1] = d.at<float>(1, 0);
	coeff[2] = d.at<float>(2, 0) / norm_f;
	coeff[3] = d.at<float>(3, 0) / norm_f / norm_f;

	//printf("%f %f %f\n", coeff[0], coeff[1], coeff[2]); 
	//cvmMul(a, d, e);
	e = a*d;
	//err = abs(cvmGet(e,0,0) - cvmGet(b,0,0))+abs(cvmGet(e,1,0) - cvmGet(b,1,0))+abs(cvmGet(e,2,0) - cvmGet(b,2,0));
	err = 0;

	for (int i = 0; i < n; i++) {
		points[i].x = (float)cvRound(points[i].x * norm_f);
		points[i].y = (float)cvRound(points[i].y * norm_f);
	}

	return err;
}
