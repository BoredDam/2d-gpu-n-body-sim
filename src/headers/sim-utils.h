#include <CL/cl.h>
#include <stdio.h>
#define OUTPUTS_PATH "./outputs/"
#define BENCHMARKS_PATH "./tests/"
#define GALAXIES_PATH "./galaxies/"

void write_frame_on_disk(const int count, const cl_float2 *bodies, const char *sim_name, const int time) {
    /* 
    this function is SHIT.
    it does work, but its just too much overhead...
    gotta find a smarter way to handle 
    output file creation :')
    */
    FILE *fp;
    char file_name[512];
    sprintf(file_name, "./outputs/%s/%s_frame_%d.csv", sim_name, sim_name, time);
    fp = fopen(file_name, "w+");

    fprintf(fp, "x,y\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%f,%f\n", bodies[i].x, bodies[i].y);
    }
    fclose(fp); 
}


void write_bounding_box_on_disk(
    const int count, 
    const cl_float2 *centers, 
    const cl_float *half_size, 
    const char *sim_name, 
    const int time
) {
    /* 
    another shitty function to handle writing frames on disk:/
    */
    FILE *fp;
    char file_name[512];
    sprintf(file_name, "./outputs/%s/%s_frame_%d.csv", sim_name, sim_name, time);
    fp = fopen(file_name, "w+");

    fprintf(fp, "x,y,r\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%f,%f,%f\n", centers[i].x, centers[i].y, half_size[i]);
    }
    fclose(fp); 
}


void write_naive_stats_on_disk(
    const double update_pos_ms,
    const double update_vel_ms,
    const double update_acc_ms,
    const int body_count, 
    const char *sim_name
) {
    /* 
    another shitty function to handle writing frames on disk:/
    */
    FILE *fp;
    char file_name[512];
    sprintf(file_name, "./tests/%s/%s.csv", sim_name, sim_name);
    fp = fopen(file_name, "a+");
    fprintf(fp, "%f,%f,%f,%d\n", update_pos_ms, update_vel_ms, update_acc_ms, body_count);
    fclose(fp); 
}


void write_bh_stats_on_disk(
    const double reduction_min_ms, 
    const double reduction_max_ms, 
    const double reset_init_tree_ms, 
    const double build_tree_ms, 
    const double summarize_tree_ms, 
    const double compute_acc_ms, 
    const double update_vel_ms, 
    const double update_pos_ms,
    const int body_count, 
    const char *sim_name
) {
    /* 
    another shitty function to handle writing data on disk:/
    */
    FILE *fp;
    char file_name[512];
    sprintf(file_name, "./tests/%s/%s.csv", sim_name, sim_name);
    fp = fopen(file_name, "a+");
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%d\n",
       reduction_min_ms, reduction_max_ms, reset_init_tree_ms, build_tree_ms, 
       summarize_tree_ms, compute_acc_ms, update_vel_ms, update_pos_ms, body_count);
    fclose(fp); 
}


void write_slidingwindow_stats_on_disk(
    const double reduction_k1, 
    const double reduction_k2,
    const int lws, 
    const int nwg_cu, 
    const int body_count, 
    const char *sim_name
) {
    /* 
    another shitty function to handle writing data on disk:/
    */
    FILE *fp;
    char file_name[512];
    sprintf(file_name, "./tests/%s/%s.csv", sim_name, sim_name);
    fp = fopen(file_name, "a+");
    fprintf(fp, "%f,%f,%f,%d,%d,%d\n",
        reduction_k1, reduction_k2, reduction_k1+reduction_k2,
        lws, nwg_cu, body_count
    );
    fclose(fp); 
}