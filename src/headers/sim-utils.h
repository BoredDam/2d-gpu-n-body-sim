#include <CL/cl.h>
#include <stdio.h>

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