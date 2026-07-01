#include <CL/cl.h>
#include <stdio.h>

void write_frame_on_disk(const int count, const cl_float8 *bodies, const char *sim_name, const int time) {
    /* 
    this function is SHIT.
    it does work, but its just too much overhead...
    gotta find a smarter way to handle 
    output file creation :')
    */
    FILE *fptr;
    char file_name[512];
    sprintf(file_name, "./outputs/%s/%s_frame_%d.csv", sim_name, sim_name, time);
    fptr = fopen(file_name, "w+");

    fprintf(fptr, "x,y,z\n");
    for (int i = 0; i < count; i++) {
        fprintf(fptr, "%f,%f,%f\n", bodies[i].s1, bodies[i].s2, bodies[i].s3);
    }
    fclose(fptr); 
}