#include "./headers/ocl_boiler.h"
#include "./headers/sim-utils.h"
#include <sys/stat.h>
#include <math.h>
#include <limits.h>
#include <linux/limits.h>
#include <stdbool.h>

#define DELTA_TIME 0.02f
#define CENTER_DISTANCE 10
#define SEED 42
#define MAX_TREE_DEPTH 32
#define NODE_PER_BODY 8
#define CHILDREN 4

/* prototipi */

cl_event compute_acc_walk_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem body_pos,
    cl_mem body_mass,
    cl_mem body_acc,
    cl_mem cell_mass,
    cl_mem cell_center,
    cl_mem cell_half_size,
    cl_mem cell_center_of_mass,
    cl_mem cell_children,
    float theta_squared,
    unsigned int body_count
);

cl_event update_pos_run(
    cl_command_queue que, 
    cl_kernel k, 
    cl_mem body_pos, 
    cl_mem body_vel,
    unsigned int body_count,
    cl_float delta_time
);

cl_event update_vel_run(
    cl_command_queue que, 
    cl_kernel k,
    cl_mem body_vel,
    cl_mem body_acc,
    cl_float delta_time,
    unsigned int body_count
);

cl_event reduce_minmax_lmem_sliding_k1_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem pos,
    cl_mem res_min,
    cl_mem res_max,
    cl_mem bounding_box,
    unsigned int body_count_nhex,
    int nwg,
    cl_int lws_cli
);

cl_event reduce_minmax_lmem_sliding_k2_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem res_min,
    cl_mem res_max,
    cl_mem bounding_box,
    unsigned int group_count,
    cl_int lws_cli
);

cl_event reset_init_tree_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem cell_children,
    cl_mem cell_center,
    cl_mem cell_half_size,
    cl_mem cell_mass,
    cl_mem bounding_box,
    unsigned int max_cells
);

cl_event build_tree_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem body_pos,
    cl_mem cell_children,
    cl_mem cell_center,
    cl_mem cell_half_size,
    unsigned int body_count,
    unsigned int max_cells
);

cl_event summarize_tree_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem body_pos,
    cl_mem body_mass,
    cl_mem cell_mass,
    cl_mem cell_center_of_mass,
    cl_mem cell_children,
    unsigned int body_count,
    unsigned int max_cells
);


int main(int argc, char *argv[]) {
    
    if (argc != 9) {
        printf("correct usage: %s, [body count], [iterations], [config-name], " 
               "[simulation-name], [theta], [false: no output, true: output], "
               "[lws], [nwg cu]\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int body_count = atoi(argv[1]);
    if (body_count < 1) {
        printf("body count must be at least 1\n");
        return EXIT_FAILURE;
    }

    if (body_count % 8) {
        printf("body count must be a multiple of 8\n");
        return EXIT_FAILURE;
    }

    unsigned int iterations = atoi(argv[2]);
    if (iterations < 1) {
        printf("iterations must be at least 1\n");
        return EXIT_FAILURE;
    }

    char *galaxy_name = argv[3];
    char *sim_name = argv[4];
    float theta = atof(argv[5]);
    float theta_squared = theta * theta;
    bool wants_output = atoi(argv[6]);

    int lws = atoi(argv[7]);
	if (lws < 1) {
		printf("lws must be at least 1\n");
		return EXIT_FAILURE;
	}
    
	int nwg_cu = atoi(argv[8]);
	if (nwg_cu < 1) {
		printf("nwg cu must be at least 1\n");
		return EXIT_FAILURE;
	}

    /*openCL setup*/
    cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("src/kernels/barnes_hut_v2.ocl", ctx, d);
    
    cl_int err;
    cl_kernel reduction_bbox_k1 = clCreateKernel(prog, "reduce_minmax_lmem_sliding_k1", &err);
    ocl_check(err, "clCreateKernel failed on reduction_bbox_k1");

    cl_kernel reduction_bbox_k2 = clCreateKernel(prog, "reduce_minmax_lmem_sliding_k2", &err);
    ocl_check(err, "clCreateKernel failed on reduction_bbox_k2");

    cl_kernel reset_init_tree_k = clCreateKernel(prog, "reset_init_tree", &err );
    ocl_check(err, "clCreateKernel failed on reset_init_tree");

    cl_kernel serial_build_tree_k = clCreateKernel(prog, "serial_build_tree", &err);
    ocl_check(err, "clCreateKernel failed on serial_build_tree");

    cl_kernel summarize_tree_k = clCreateKernel(prog, "summarize_tree", &err);
    ocl_check(err, "clCreateKernel failed on summarize_tree");

    cl_kernel compute_acc_walk_k = clCreateKernel(prog, "compute_acc_walk", &err);
    ocl_check(err, "clCreateKernel failed on update_pos");

    cl_kernel update_vel_k = clCreateKernel(prog, "update_vel", &err);
    ocl_check(err, "clCreateKernel failed on update_pos");

    cl_kernel update_pos_k = clCreateKernel(prog, "update_pos", &err);
    ocl_check(err, "clCreateKernel failed on update_pos");
    
    /*setting up the configuration*/
    cl_float2 *body_pos = malloc(sizeof(cl_float2) * body_count);
    cl_float2 *body_vel = malloc(sizeof(cl_float2) * body_count);
    cl_float *body_mass = malloc(sizeof(cl_float) * body_count);

    if (!body_pos || !body_vel || !body_mass) {
        free(body_pos);
        free(body_vel);
        free(body_mass);
        return EXIT_FAILURE;
    }

    size_t body_pos_buffer_size = sizeof(cl_float2) * body_count;
    size_t body_vel_buffer_size = sizeof(cl_float2) * body_count;
    size_t body_acc_buffer_size = sizeof(cl_float2) * body_count;
    size_t body_mass_buffer_size = sizeof(cl_float) * body_count;

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
    free(body_mass);

    unsigned int max_cells = body_count * NODE_PER_BODY + 1;
    size_t children_buffer_size = sizeof(cl_int) * (max_cells) * CHILDREN;
    size_t cell_2dfloat_buffer_size = sizeof(cl_float2) * (body_count * NODE_PER_BODY + 1);
    size_t cell_1dfloat_buffer_size = sizeof(cl_float) * (body_count * NODE_PER_BODY + 1);
    size_t reduction_buffer_size = sizeof(cl_float2) * body_count;

    /* tutti i dati necessari per la riduzione parallela con sliding window local memory della v2*/
    cl_int cu;
	ocl_check(
        clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL), 
        "get max compute_units"
    );

	int nwg = nwg_cu * cu;
	if (nwg > 1) {
		nwg = round_mul_up(nwg, 16);
	}
	printf("%d wg per CU, %d CU = %d work-groups\n", nwg_cu, cu, nwg);


    /*DEBUG*/
    cl_float2 bounding_box_host[2];

    /* INSERIRE QUI TUTTA LA ROBA DA ALLOCARE DIRETTAMENTE 
    NELLA GPU, QUINDI QUAD-TREE, ARRAY FORZA E SIMILI*/

    cl_mem bounding_box_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE,
        sizeof(cl_float2) * 2,
        NULL,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on bounding_box_mem");

    cl_mem body_acc_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE,
        body_acc_buffer_size,
        NULL,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on body_acc_mem");

    err = clEnqueueFillBuffer(
        que,
        body_acc_mem,
        &(cl_float2){0.0f, 0.0f},
        sizeof(cl_float2),
        0,               
        body_acc_buffer_size,
        0, 
        NULL, 
        NULL
    );
    ocl_check(err, "clEnqueueFillBuffer failed on body_acc_mem");

    cl_mem reduction_buffer1 = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE,
        reduction_buffer_size,
        NULL,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on reduction_buffer1");

    cl_mem reduction_buffer2 = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE,
        reduction_buffer_size,
        NULL,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on reduction_buffer2");
    
    cl_mem cell_children_mem = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE, children_buffer_size, NULL, &err
    );
    ocl_check(err, "clCreateBuffer cell_children");

    cl_mem cell_center_mem = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE, cell_2dfloat_buffer_size, NULL, &err
    );
    ocl_check(err, "clCreateBuffer cell_center");

    cl_mem cell_half_size_mem = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE, cell_1dfloat_buffer_size, NULL, &err
    );
    ocl_check(err, "clCreateBuffer cell_half_size");

    cl_mem cell_center_of_mass_mem = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE, cell_2dfloat_buffer_size, NULL, &err
    );
    ocl_check(err, "clCreateBuffer cell_center_of_mass");

    cl_mem cell_mass_mem = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE, cell_1dfloat_buffer_size, NULL, &err
    );
    ocl_check(err, "clCreateBuffer cell_mass");

    unsigned int reduction_count = ceil(log2(body_count));

    cl_event compute_acc_walk_event[iterations + 1],
             update_pos_event[iterations],
             update_vel_event[iterations + 1],
             reduction_k1_event[iterations + 1],
             reduction_k2_event[iterations + 1],
             reset_init_tree_event[iterations + 1],
             build_tree_event[iterations + 1],
             summarize_tree_event[(iterations + 1) * MAX_TREE_DEPTH];

    if (wants_output) {
        char outputs_path_name[PATH_MAX + 1] = OUTPUTS_PATH;
        strcat(outputs_path_name, sim_name);
        mkdir(outputs_path_name, S_IRWXU);
    }

    char outputs_path_name[PATH_MAX + 1] = BENCHMARKS_PATH;
    strcat(outputs_path_name, sim_name);
    mkdir(outputs_path_name, S_IRWXU);


    /*actual sim*/


    /*(leapfrog halfstep) finding both minimum and maximum x and y 
      in the particle system to find the bounding box coordinates*/
    
    unsigned int body_count_nhex = body_count / 8;
    reduction_k1_event[iterations] = reduce_minmax_lmem_sliding_k1_run(
        que,
        reduction_bbox_k1,
        body_pos_mem,
        reduction_buffer1,
        reduction_buffer2,
        bounding_box_mem,
        body_count_nhex,
        nwg,
        lws
    );

    clWaitForEvents(1, &reduction_k1_event[iterations]);

    reduction_k2_event[iterations] = reduce_minmax_lmem_sliding_k2_run(
        que,
        reduction_bbox_k2,
        reduction_buffer1,
        reduction_buffer2,
        bounding_box_mem,
        nwg,
        lws
    );
    clWaitForEvents(1, &reduction_k2_event[iterations]);

    /* DEBUG */
    err = clEnqueueReadBuffer(
        que,
        bounding_box_mem,
        CL_TRUE,
        0,
        sizeof(bounding_box_host),
        bounding_box_host,
        0,
        NULL,
        NULL
    );
    ocl_check(err, "clEnqueueReadBuffer bounding_box");

    printf(
        "bounding box:\n"
        "min = (%f, %f)\n"
        "max = (%f, %f)\n",
        bounding_box_host[0].x,
        bounding_box_host[0].y,
        bounding_box_host[1].x,
        bounding_box_host[1].y
    );

    /*resetting the indexes and values inside the
      quadtree arrays to the default value*/
    reset_init_tree_event[iterations] = reset_init_tree_run(
        que,
        reset_init_tree_k,
        cell_children_mem,
        cell_center_mem,
        cell_half_size_mem,
        cell_mass_mem,
        bounding_box_mem,
        max_cells
    );
    clWaitForEvents(1, &reset_init_tree_event[iterations]);

    /*serial :( tree building kernel*/
    build_tree_event[iterations] = build_tree_run(
        que,
        serial_build_tree_k,
        body_pos_mem,
        cell_children_mem,
        cell_center_mem,
        cell_half_size_mem,
        body_count,
        max_cells
    );
    clWaitForEvents(1, &build_tree_event[iterations]);

    /*propagation of the masses and the center of masses
      from the leaves to the root. one step for each
      level of the tree.*/
    #pragma unroll MAX_TREE_DEPTH
    for (int i = 0; i < MAX_TREE_DEPTH; i++) {
        summarize_tree_event[iterations * MAX_TREE_DEPTH + i] = summarize_tree_run(
            que,
            summarize_tree_k,
            body_pos_mem,
            body_mass_mem,
            cell_mass_mem,
            cell_center_of_mass_mem,
            cell_children_mem,
            body_count,
            max_cells
        );
        clWaitForEvents(1, &summarize_tree_event[iterations * MAX_TREE_DEPTH + i]);
    }
    
    /*because of my thesis i love the term "walk" when talking
     about traversing graphs and trees. anyway, in this step 
     we compute the acceleration for each body using the quadtree
     we just built*/
    compute_acc_walk_event[iterations] = compute_acc_walk_run(
        que,
        compute_acc_walk_k,
        body_pos_mem,
        body_mass_mem,
        body_acc_mem,  
        cell_mass_mem,
        cell_center_mem,
        cell_half_size_mem,
        cell_center_of_mass_mem,
        cell_children_mem,
        theta_squared,
        body_count
    );
    clWaitForEvents(1, &compute_acc_walk_event[iterations]);
    
    update_vel_event[iterations] = update_vel_run(
        que,
        update_vel_k,
        body_vel_mem,
        body_acc_mem,
        DELTA_TIME/2,
        body_count
    );
    clWaitForEvents(1, &update_vel_event[iterations]);
    
    /*rest of the simulation*/
    if (wants_output) {
        for (int i = 0; i < iterations; i++) {
            update_pos_event[i] = update_pos_run(
                que,
                update_pos_k,
                body_pos_mem,
                body_vel_mem,
                body_count,
                DELTA_TIME
            );
            clWaitForEvents(1, &update_pos_event[i]);
        
            body_pos = clEnqueueMapBuffer(
                que, 
                body_pos_mem, 
                CL_TRUE, 
                CL_MAP_READ, 
                0, 
                body_pos_buffer_size, 
                0, 
                NULL, 
                NULL, 
                &err
            );
            ocl_check(err, "enqueueMapBuffer failed");

            write_frame_on_disk(body_count, body_pos, sim_name, i);

            err = clEnqueueUnmapMemObject(
                que,
                body_pos_mem,
                body_pos,
                0, 
                NULL, 
                NULL
            );
            ocl_check(err, "enqueueUnmapObject failed");

            reduction_k1_event[i] = reduce_minmax_lmem_sliding_k1_run(
                que,
                reduction_bbox_k1,
                body_pos_mem,
                reduction_buffer1,
                reduction_buffer2,
                bounding_box_mem,
                body_count_nhex,
                nwg,
                lws
            );
            clWaitForEvents(1, &reduction_k1_event[i]);

            reduction_k2_event[i] = reduce_minmax_lmem_sliding_k2_run(
                que,
                reduction_bbox_k2,
                reduction_buffer1,
                reduction_buffer2,
                bounding_box_mem,
                nwg,
                lws
            );
            clWaitForEvents(1, &reduction_k2_event[i]);
        
            reset_init_tree_event[i] = reset_init_tree_run(
                que,
                reset_init_tree_k,
                cell_children_mem,
                cell_center_mem,
                cell_half_size_mem,
                cell_mass_mem,
                bounding_box_mem,
                max_cells
            );
            clWaitForEvents(1, &reset_init_tree_event[i]);
        
            build_tree_event[i] = build_tree_run(
                que,
                serial_build_tree_k,
                body_pos_mem,
                cell_children_mem,
                cell_center_mem,
                cell_half_size_mem,
                body_count,
                max_cells
            );
            clWaitForEvents(1, &build_tree_event[i]);
        
            #pragma unroll MAX_TREE_DEPTH
            for (int j = 0; j < MAX_TREE_DEPTH; j++) {
                summarize_tree_event[i * MAX_TREE_DEPTH + j] = summarize_tree_run(
                    que,
                    summarize_tree_k,
                    body_pos_mem,
                    body_mass_mem,
                    cell_mass_mem,
                    cell_center_of_mass_mem,
                    cell_children_mem,
                    body_count,
                    max_cells
                );
                clWaitForEvents(1, &summarize_tree_event[i * MAX_TREE_DEPTH + j]);
            }
        
            compute_acc_walk_event[i] = compute_acc_walk_run(
                que,
                compute_acc_walk_k,
                body_pos_mem,
                body_mass_mem,
                body_acc_mem,  
                cell_mass_mem,
                cell_center_mem,
                cell_half_size_mem,
                cell_center_of_mass_mem,
                cell_children_mem,
                theta_squared,
                body_count
            );
            clWaitForEvents(1, &compute_acc_walk_event[i]);
        
            update_vel_event[i] = update_vel_run(
                que,
                update_vel_k,
                body_vel_mem,
                body_acc_mem,
                DELTA_TIME,
                body_count
            );
            clWaitForEvents(1, &update_vel_event[i]);
        }
    } else {

        for (int i = 0; i < iterations; i++) {
            update_pos_event[i] = update_pos_run(
                que,
                update_pos_k,
                body_pos_mem,
                body_vel_mem,
                body_count,
                DELTA_TIME
            );
            clWaitForEvents(1, &update_pos_event[i]);
        
            reduction_k1_event[i] = reduce_minmax_lmem_sliding_k1_run(
                que,
                reduction_bbox_k1,
                body_pos_mem,
                reduction_buffer1,
                reduction_buffer2,
                bounding_box_mem,
                body_count_nhex,
                nwg,
                lws
            );
            clWaitForEvents(1, &reduction_k1_event[i]);

            reduction_k2_event[i] = reduce_minmax_lmem_sliding_k2_run(
                que,
                reduction_bbox_k2,
                reduction_buffer1,
                reduction_buffer2,
                bounding_box_mem,
                nwg,
                lws
            );
            clWaitForEvents(1, &reduction_k2_event[i]);

            reset_init_tree_event[i] = reset_init_tree_run(
                que,
                reset_init_tree_k,
                cell_children_mem,
                cell_center_mem,
                cell_half_size_mem,
                cell_mass_mem,
                bounding_box_mem,
                max_cells
            );
            clWaitForEvents(1, &reset_init_tree_event[i]);
        
            build_tree_event[i] = build_tree_run(
                que,
                serial_build_tree_k,
                body_pos_mem,
                cell_children_mem,
                cell_center_mem,
                cell_half_size_mem,
                body_count,
                max_cells
            );
            clWaitForEvents(1, &build_tree_event[i]);
        
            #pragma unroll MAX_TREE_DEPTH
            for (int j = 0; j < MAX_TREE_DEPTH; j++) {
                summarize_tree_event[i * MAX_TREE_DEPTH + j] = summarize_tree_run(
                    que,
                    summarize_tree_k,
                    body_pos_mem,
                    body_mass_mem,
                    cell_mass_mem,
                    cell_center_of_mass_mem,
                    cell_children_mem,
                    body_count,
                    max_cells
                );
                clWaitForEvents(1, &summarize_tree_event[i * MAX_TREE_DEPTH + j]);
            }
        
            compute_acc_walk_event[i] = compute_acc_walk_run(
                que,
                compute_acc_walk_k,
                body_pos_mem,
                body_mass_mem,
                body_acc_mem,  
                cell_mass_mem,
                cell_center_mem,
                cell_half_size_mem,
                cell_center_of_mass_mem,
                cell_children_mem,
                theta_squared,
                body_count
            );
            clWaitForEvents(1, &compute_acc_walk_event[i]);
        
            update_vel_event[i] = update_vel_run(
                que,
                update_vel_k,
                body_vel_mem,
                body_acc_mem,
                DELTA_TIME,
                body_count
            );
            clWaitForEvents(1, &update_vel_event[i]);
        }
    }
    
    clFinish(que);


    double compute_acc_ms = 0,
           update_pos_ms = 0,
           update_vel_ms = 0,
           reduction_k1_ms = 0,
           reduction_k2_ms = 0,
           reset_init_tree_ms = 0,
           build_tree_ms = 0,
           summarize_tree_ms = 0;

    for (int i = 0; i < iterations; i++) {
        compute_acc_ms += runtime_ms(compute_acc_walk_event[i]);
        update_pos_ms += runtime_ms(update_pos_event[i]);
        update_vel_ms += runtime_ms(update_vel_event[i]);
        reduction_k1_ms += runtime_ms(reduction_k1_event[i]);
        reduction_k2_ms += runtime_ms(reduction_k2_event[i]);
        reset_init_tree_ms += runtime_ms(reset_init_tree_event[i]);
        build_tree_ms += runtime_ms(build_tree_event[i]);
        for (int j = 0; j < MAX_TREE_DEPTH; j++) {
            summarize_tree_ms += runtime_ms(summarize_tree_event[i * MAX_TREE_DEPTH + j]);
        }
    }
    
    compute_acc_ms += runtime_ms(compute_acc_walk_event[iterations]);
    update_vel_ms += runtime_ms(update_vel_event[iterations]);
    reduction_k1_ms += runtime_ms(reduction_k1_event[iterations]);
    reduction_k2_ms += runtime_ms(reduction_k2_event[iterations]);
    reset_init_tree_ms += runtime_ms(reset_init_tree_event[iterations]);
    build_tree_ms += runtime_ms(build_tree_event[iterations]);
    for (int j = 0; j < MAX_TREE_DEPTH; j++) {
        summarize_tree_ms += runtime_ms(summarize_tree_event[iterations * MAX_TREE_DEPTH + j]);
    }


    printf("TIMES:\n\nreduction_k1: %fms\nreduction_k2: %fms\nreset_init: %fms\n"
           "build: %fms\nsummarize: %fms\ncompute_acc: %fms\nupdate_vel: %fms\n"
           "update_pos: %fms\n",
           reduction_k1_ms, reduction_k2_ms, reset_init_tree_ms, build_tree_ms, 
           summarize_tree_ms, compute_acc_ms, update_vel_ms, update_pos_ms
        );

    write_bh_stats_on_disk(reduction_k1_ms, reduction_k2_ms, reset_init_tree_ms, build_tree_ms,summarize_tree_ms, 
                           compute_acc_ms, update_vel_ms, update_pos_ms, body_count, sim_name);

    clReleaseMemObject(body_pos_mem);
    clReleaseMemObject(body_vel_mem);
    clReleaseMemObject(body_mass_mem);
    clReleaseMemObject(body_acc_mem);
    clReleaseMemObject(cell_center_mem);
    clReleaseMemObject(cell_mass_mem);
    clReleaseMemObject(cell_center_of_mass_mem);
    clReleaseMemObject(cell_half_size_mem);
    clReleaseMemObject(cell_children_mem);
    clReleaseMemObject(bounding_box_mem);
    clReleaseMemObject(reduction_buffer1);
    clReleaseMemObject(reduction_buffer2);
    
    clReleaseKernel(reduction_bbox_k1);
    clReleaseKernel(reduction_bbox_k2);
    clReleaseKernel(reset_init_tree_k);
    clReleaseKernel(serial_build_tree_k);
    clReleaseKernel(summarize_tree_k);
    clReleaseKernel(compute_acc_walk_k);
    clReleaseKernel(update_vel_k);
    clReleaseKernel(update_pos_k);

    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    
    return EXIT_SUCCESS;
}



cl_event compute_acc_walk_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem body_pos,
    cl_mem body_mass,
    cl_mem body_acc,
    cl_mem cell_mass,
    cl_mem cell_center,
    cl_mem cell_half_size,
    cl_mem cell_center_of_mass,
    cl_mem cell_children,
    float theta_squared,
    unsigned int body_count
) {
    cl_event event;
    const size_t gws[1] = {round_mul_up(body_count, 32)};

    cl_int err;
    cl_uint arg = 0;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_pos),
        &body_pos
    );
    ocl_check(err, "clSetKernelArg body_pos");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_mass),
        &body_mass
    );
    ocl_check(err, "clSetKernelArg body_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_acc),
        &body_acc
    );
    ocl_check(err, "clSetKernelArg body_acc");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_mass),
        &cell_mass
    );
    ocl_check(err, "clSetKernelArg cell_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center),
        &cell_center
    );
    ocl_check(err, "clSetKernelArg cell_center");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_half_size),
        &cell_half_size
    );
    ocl_check(err, "clSetKernelArg cell_half_size");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center_of_mass),
        &cell_center_of_mass
    );
    ocl_check(err, "clSetKernelArg cell_center_of_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_children),
        &cell_children
    );
    ocl_check(err, "clSetKernelArg cell_children");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(theta_squared),
        &theta_squared
    );
    ocl_check(err, "clSetKernelArg theta_squared");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_count),
        &body_count
    );
    ocl_check(err, "clSetKernelArg body_count");
    ++arg;

    err = clEnqueueNDRangeKernel(
        que,
        k,
        1,
        NULL,
        gws,
        NULL,
        0,
        NULL,
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel compute_acc_walk");

    return event;
}


cl_event update_pos_run(
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
    cl_uint arg = 0;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_pos), &body_pos);
    ocl_check(err,"clSetKernelArg body_pos");
    ++arg;
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_vel), &body_vel);
    ocl_check(err,"clSetKernelArg body_vel");
    ++arg;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(delta_time), &delta_time);
    ocl_check(err,"clSetKernelArg update_pos delta_time");
    ++arg;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg body_count");
    ++arg;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


cl_event update_vel_run(
    cl_command_queue que, 
    cl_kernel k,
    cl_mem body_vel,
    cl_mem body_acc,
    cl_float delta_time,
    unsigned int body_count
) { 
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;

    cl_uint arg = 0;
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_vel), 
        &body_vel
    );
    ocl_check(err,"clSetKernelArg body_vel");
    ++arg;
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_acc), 
        &body_acc
    );
    ocl_check(err,"clSetKernelArg body_mass");
    ++arg;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(delta_time), 
        &delta_time
    );
    ocl_check(err,"clSetKernelArg update_pos delta_time");
    ++arg;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_count), 
        &body_count
    );
    ocl_check(err,"clSetKernelArg body_count");

    err = clEnqueueNDRangeKernel(
        que, 
        k, 
        1, 
        NULL, 
        gws, 
        NULL, 
        0, 
        NULL, 
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel");

    return event;
}


cl_event reduce_minmax_lmem_sliding_k1_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem pos,
    cl_mem res_min,
    cl_mem res_max,
    cl_mem bounding_box,
    unsigned int body_count_nhex,
    int nwg,
    cl_int lws_in
) {
    cl_event event;
    cl_int err;

    const size_t lws[1] = { lws_in };
    const size_t gws[1] = { nwg * lws[0] };

    cl_uint arg = 0;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(pos),
        &pos
    );
    ocl_check(err, "clSetKernelArg reduction k1 pos");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(res_min),
        &res_min
    );
    ocl_check(err, "clSetKernelArg reduction k1 res_min");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(res_max),
        &res_max
    );
    ocl_check(err, "clSetKernelArg reduction k1 res_max");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(bounding_box),
        &bounding_box
    );
    ocl_check(err, "clSetKernelArg reduction k1 bounding_box");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_count_nhex),
        &body_count_nhex
    );
    ocl_check(err, "clSetKernelArg reduction k1 body_count_nhex");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        lws[0] * sizeof(cl_float2),
        NULL
    );
    ocl_check(err, "clSetKernelArg reduction k1 local_min");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        lws[0] * sizeof(cl_float2),
        NULL
    );
    ocl_check(err, "clSetKernelArg reduction k1 local_max");

    err = clEnqueueNDRangeKernel(
        que,
        k,
        1,
        NULL,
        gws,
        lws,
        0,
        NULL,
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel reduce_minmax_lmem_sliding_k1_run");

    return event;
}


cl_event reduce_minmax_lmem_sliding_k2_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem res_min,
    cl_mem res_max,
    cl_mem bounding_box,
    unsigned int group_count,
    cl_int lws_cli
) {
    cl_event event;
    cl_int err;

    const size_t lws[1] = { lws_cli };
    const size_t gws[1] = { lws[0] };

    cl_uint arg = 0;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(res_min),
        &res_min
    );
    ocl_check(err, "clSetKernelArg reduction k2 res_min");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(res_max),
        &res_max
    );
    ocl_check(err, "clSetKernelArg reduction k2 res_max");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(bounding_box),
        &bounding_box
    );
    ocl_check(err, "clSetKernelArg reduction k2 bounding_box");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(group_count),
        &group_count
    );
    ocl_check(err, "clSetKernelArg reduction k2 group_count");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        lws[0] * sizeof(cl_float2),
        NULL
    );
    ocl_check(err, "clSetKernelArg reduction k2 local_min");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        lws[0] * sizeof(cl_float2),
        NULL
    );
    ocl_check(err, "clSetKernelArg reduction k2 local_max");

    err = clEnqueueNDRangeKernel(
        que,
        k,
        1,
        NULL,
        gws,
        lws,
        0,
        NULL,
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel reduce_minmax_lmem_sliding_k2_run");

    return event;
}


cl_event reset_init_tree_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem cell_children,
    cl_mem cell_center,
    cl_mem cell_half_size,
    cl_mem cell_mass,
    cl_mem bounding_box,
    unsigned int max_cells
) {
    cl_event event;
    cl_int err;

    cl_uint arg = 0;

    const size_t gws[1] = { round_mul_up(max_cells, 32) };

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(cell_children), 
        &cell_children
    );
    ocl_check(err, "clSetKernelArg cell_children");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center),
        &cell_center
    );
    ocl_check(err, "clSetKernelArg cell_center");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_half_size),
        &cell_half_size
    );
    ocl_check(err, "clSetKernelArg cell_half_size");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_mass),
        &cell_mass
    );
    ocl_check(err, "clSetKernelArg cell_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(bounding_box),
        &bounding_box
    );
    ocl_check(err, "clSetKernelArg bounding_box");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(max_cells),
        &max_cells
    );
    ocl_check(err, "clSetKernelArg max_cells");
    ++arg;


    err = clEnqueueNDRangeKernel(
        que,
        k,
        1,
        NULL,
        gws,
        NULL,
        0,
        NULL,
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel reset_init_tree_run");

    return event;
}


cl_event build_tree_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem body_pos,
    cl_mem cell_children,
    cl_mem cell_center,
    cl_mem cell_half_size,
    unsigned int body_count,
    unsigned int max_cells
) {
    cl_event event;
    cl_int err;

    cl_uint arg = 0;

    const size_t gws[1] = { 1 };
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_pos), 
        &body_pos
    );
    ocl_check(err, "clSetKernelArg body_pos");
    ++arg;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(cell_children), 
        &cell_children
    );
    ocl_check(err, "clSetKernelArg cell_children");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center),
        &cell_center
    );
    ocl_check(err, "clSetKernelArg cell_center");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_half_size),
        &cell_half_size
    );
    ocl_check(err, "clSetKernelArg cell_half_size");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_count),
        &body_count
    );
    ocl_check(err, "clSetKernelArg body_count");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(max_cells),
        &max_cells
    );
    ocl_check(err, "clSetKernelArg max_cells");
    ++arg;


    err = clEnqueueNDRangeKernel(
        que,
        k,
        1,
        NULL,
        gws,
        NULL,
        0,
        NULL,
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel build_tree_run");

    return event;
}


cl_event summarize_tree_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem body_pos,
    cl_mem body_mass,
    cl_mem cell_mass,
    cl_mem cell_center_of_mass,
    cl_mem cell_children,
    unsigned int body_count,
    unsigned int max_cells
) {
    cl_event event;
    cl_int err;

    cl_uint arg = 0;

    const size_t gws[1] = {
        round_mul_up(max_cells, 32)
    };

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_pos),
        &body_pos
    );
    ocl_check(err, "clSetKernelArg body_pos");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_mass),
        &body_mass
    );
    ocl_check(err, "clSetKernelArg body_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_mass),
        &cell_mass
    );
    ocl_check(err, "clSetKernelArg cell_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center_of_mass),
        &cell_center_of_mass
    );
    ocl_check(err, "clSetKernelArg cell_center_of_mass");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_children),
        &cell_children
    );
    ocl_check(err, "clSetKernelArg cell_children");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_count),
        &body_count
    );
    ocl_check(err, "clSetKernelArg body_count");
    ++arg;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(max_cells),
        &max_cells
    );
    ocl_check(err, "clSetKernelArg max_cells");
    ++arg;

    err = clEnqueueNDRangeKernel(
        que,
        k,
        1,
        NULL,
        gws,
        NULL,
        0,
        NULL,
        &event
    );
    ocl_check(err, "clEnqueueNDRangeKernel summarize_tree_run");

    return event;
}
