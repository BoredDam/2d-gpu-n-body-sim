#include "./headers/ocl_boiler.h"
#include "./headers/sim-utils.h"
#include <sys/stat.h>
#include <limits.h>
#include <linux/limits.h>

#define DELTA_TIME 0.02f
#define CENTER_DISTANCE 10
#define GALAXIES_PATH "./galaxies/"
#define OUTPUTS_PATH "./outputs/"
#define SEED 42



cl_event update_acc_run_k(
    cl_command_queue que, 
    cl_kernel k, 
    cl_mem body_pos,
    cl_mem body_acc,
    cl_mem body_mass,
    unsigned int body_count
) {
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(k, arg_index, sizeof(body_pos), &body_pos);
    ocl_check(err,"clSetKernelArg body_pos");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_acc), &body_acc);
    ocl_check(err,"clSetKernelArg body_acc");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_mass), &body_mass);
    ocl_check(err,"clSetKernelArg body_mass");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg body_count");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


cl_event update_pos_run_k(
    cl_command_queue que, 
    cl_kernel k, 
    cl_mem body_pos, 
    cl_mem body_vel,
    unsigned int body_count,
    cl_float delta_time
) {
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(k, arg_index, sizeof(body_pos), &body_pos);
    ocl_check(err,"clSetKernelArg body_pos");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_vel), &body_vel);
    ocl_check(err,"clSetKernelArg body_vel");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(delta_time), &delta_time);
    ocl_check(err,"clSetKernelArg update_pos delta_time");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg body_count");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


cl_event update_vel_run_k(
    cl_command_queue que, 
    cl_kernel k,
    cl_mem body_vel,
    cl_mem body_acc,
    unsigned int body_count,
    cl_float delta_time
) { 
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;
    cl_uint arg_index = 0;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_vel), &body_vel);
    ocl_check(err,"clSetKernelArg body_vel");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_acc), &body_acc);
    ocl_check(err,"clSetKernelArg body_mass");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(delta_time), &delta_time);
    ocl_check(err,"clSetKernelArg update_pos delta_time");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg body_count");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


int main(int argc, char *argv[]) {
    
    if (argc < 5) {
        printf("correct usage: %s, [body count], [iterations], [config-name], [simulation-name]\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int body_count = atoi(argv[1]);
    if (body_count <= 0) {
        printf("body count must be at least 1\n");
        return EXIT_FAILURE;
    }

    unsigned int iterations = atoi(argv[2]);
    if (iterations <= 0) {
        printf("iterations must be at least 1\n");
        return EXIT_FAILURE;
    }

    char *galaxy_name = argv[3];
    char *sim_name = argv[4];

    /*openCL shenanigans*/
    cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("src/kernels/naive_nbody.ocl", ctx, d);
    
    cl_int err;
    cl_kernel update_acc_k = clCreateKernel(prog, "update_acc", &err);
    ocl_check(err, "clCreateKernel failed on update_acc");

    cl_kernel update_pos_k = clCreateKernel(prog, "update_pos", &err);
    ocl_check(err, "clCreateKernel failed on update_pos");

    cl_kernel update_vel_k = clCreateKernel(prog, "update_vel", &err);
    ocl_check(err, "clCreateKernel failed on update_vel");

    
    cl_float2 *body_pos = malloc(sizeof(cl_float2) * body_count);
    cl_float2 *body_vel = malloc(sizeof(cl_float2) * body_count);
    cl_float2 *body_acc = malloc(sizeof(cl_float2) * body_count);
    cl_float *body_mass = malloc(sizeof(cl_float) * body_count);

    if (!body_pos || !body_vel || !body_acc || !body_mass) {
        free(body_pos);
        free(body_vel);
        free(body_acc);
        free(body_mass);
        return EXIT_FAILURE;
    }

    size_t body_pos_buffer_size = sizeof(cl_float2) * body_count;
    size_t body_vel_buffer_size = sizeof(cl_float2) * body_count;
    size_t body_acc_buffer_size = sizeof(cl_float2) * body_count;
    size_t body_mass_buffer_size = sizeof(cl_float) * body_count;

    /*READING THE CONFIGURATION*/
    char galaxy_path_name[PATH_MAX + 1] = GALAXIES_PATH;
    strcat(galaxy_path_name, galaxy_name);

    FILE *fp = fopen(galaxy_path_name, "r");
    printf("looking for %s...\n", galaxy_path_name);
    if (fp == NULL) {
        perror("error reading the file");
        return EXIT_FAILURE;
    }
    printf("file opened succesfully\n");

    char line[128];
    float X, Y, vX, vY, mass;

    int row = 0;
    while (fgets(line, sizeof(line), fp) != NULL && row < body_count) {
        if (sscanf(line, "%f,%f,%f,%f,%f", &X, &Y, &vX, &vY, &mass) == 5) {
            body_pos[row].x = X;
            body_pos[row].y = Y;
            body_vel[row].x = vX;
            body_vel[row].y = vY;
            body_mass[row] = mass;
        } else {
            fprintf(stderr, "error reading the row no.\n%s", line);
        }
        row++;
    }

    fclose(fp);
    printf("file closed succesfully.\n");

    cl_mem body_pos_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        body_pos_buffer_size,
        body_pos,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on body_pos");

    cl_mem body_vel_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        body_vel_buffer_size,
        body_vel,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on body_vel");

    cl_mem body_acc_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        body_acc_buffer_size,
        body_acc,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on body_acc");

    cl_mem body_mass_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        body_mass_buffer_size,
        body_mass,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on body_mass");

    free(body_pos);
    free(body_vel);
    free(body_acc);
    free(body_mass);

    cl_event update_acc_event[iterations + 1], update_pos_event[iterations], update_vel_event[iterations + 1];

    update_acc_event[0] = update_acc_run_k(
        que, 
        update_acc_k, 
        body_pos_mem,
        body_acc_mem,
        body_mass_mem, 
        body_count
    );
    clWaitForEvents(1, update_acc_event);

    update_vel_event[0] = update_vel_run_k(
        que, 
        update_vel_k, 
        body_vel_mem,
        body_acc_mem,
        body_count,
        (cl_float) DELTA_TIME / 2);

    clWaitForEvents(1, update_vel_event);

    cl_event enqueue_map_buffer_event;
    char outputs_path_name[PATH_MAX + 1] = OUTPUTS_PATH;
    strcat(outputs_path_name, sim_name);
    mkdir(outputs_path_name, S_IRWXU);

    for (int i = 0; i < iterations; i++) {
        update_pos_event[i] = update_pos_run_k(
            que, 
            update_pos_k, 
            body_pos_mem, 
            body_vel_mem,
            body_count,
            (cl_float) DELTA_TIME
        );
        clWaitForEvents(1, update_pos_event + i);

        update_acc_event[i + 1] = update_acc_run_k(
            que, 
            update_acc_k, 
            body_pos_mem,
            body_acc_mem,
            body_mass_mem, 
            body_count
        );

        update_vel_event[i + 1] = update_vel_run_k(
            que, 
            update_vel_k, 
            body_vel_mem,
            body_acc_mem,
            body_count,
            (cl_float) DELTA_TIME
        );
        clWaitForEvents(1, update_vel_event + i + 1);

        body_pos = clEnqueueMapBuffer(
            que, 
            body_pos_mem, 
            CL_TRUE, 
            CL_MAP_READ, 
            0, 
            body_pos_buffer_size, 
            0, 
            NULL, 
            &enqueue_map_buffer_event, 
            &err
        );
        ocl_check(err, "enqueueMapBufferEvent failed");
        
        write_frame_on_disk(body_count, body_pos, sim_name, i);

        cl_event enqueue_unmap_event;
        err = clEnqueueUnmapMemObject(
            que,
            body_pos_mem,
            body_pos,
            0, 
            NULL, 
            &enqueue_unmap_event
        );
        ocl_check(err, "enqueueUnmapObject failed");
    }

    double time_acc_ms, time_pos_ms, time_vel_ms, time_enqueue_map_ms;
    
    time_pos_ms = total_runtime_ms(update_pos_event[0], update_pos_event[iterations - 1]);
    time_vel_ms = total_runtime_ms(update_vel_event[0], update_vel_event[iterations]);
    time_acc_ms = total_runtime_ms(update_acc_event[0], update_acc_event[iterations]);
    time_enqueue_map_ms = runtime_ms(enqueue_map_buffer_event);

    printf("TIMES:\n\nupdate_pos: %gms,\nupdate_vel: %gms,\nupdate_acc: %gms,\nenqueue_map_buffer: %gms\n",
    time_pos_ms, time_vel_ms, time_acc_ms, time_enqueue_map_ms);
    
    /*
    for (int i = 0; i < 100; i++) {
        printf("%f %f\n", body_pos[i].x, body_pos[i].y);
    }*/
    
    clReleaseMemObject(body_pos_mem);
    clReleaseMemObject(body_vel_mem);
    clReleaseMemObject(body_acc_mem);
    clReleaseMemObject(body_mass_mem);

    clReleaseKernel(update_acc_k);
    clReleaseKernel(update_pos_k);
    clReleaseKernel(update_vel_k);
    
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    
    return EXIT_SUCCESS;
}