#define _POSIX_C_SOURCE 200809L

#include "./headers/ocl_boiler.h"
#include "./headers/sim-utils.h"
#include <sys/stat.h>
#include <math.h>
#include <limits.h>
#include <linux/limits.h>
#include <stdbool.h>
#include <time.h>

#define DELTA_TIME 0.02f
#define CENTER_DISTANCE 10
#define SEED 42
#define MAX_TREE_DEPTH 32
#define NODE_PER_BODY 8
#define CHILDREN 4
#define NULL_NODE -1
#define TREE_MAPPED_BUFFERS 4



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

cl_event reduction_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem red_bufA,
    cl_mem red_bufB,
    cl_mem bounding_box,
    unsigned int is_last,
    unsigned int remaining_body_count
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


void build_tree_cpu_run(
    cl_command_queue que,
    cl_mem body_pos,
    cl_mem cell_children,
    cl_mem cell_center,
    cl_mem cell_half_size,
    unsigned int body_count,
    unsigned int max_cells,
    size_t body_pos_buffer_size,
    size_t children_buffer_size,
    size_t cell_2dfloat_buffer_size,
    size_t cell_1dfloat_buffer_size,
    cl_event *map_events,
    cl_event *unmap_events,
    struct timespec *cpu_start_event,
    struct timespec *cpu_end_event
);


void serial_build_tree_cpu(
    const cl_float2 *body_pos,
    cl_int *cell_children,
    cl_float2 *cell_center,
    cl_float *cell_half_size,
    unsigned int body_count,
    unsigned int max_cells
);


int get_quadrant(cl_float2 pos, cl_float2 center);
cl_float2 get_new_center(cl_float2 parent_center, cl_float parent_half_size, unsigned int quadrant);
int cell_to_idx(int cell_id, int body_count);
int idx_to_cell(int cell_id, int body_count);
int idx_is_cell(int ref, int body_count);
double timespec_elapsed_ms(struct timespec start, struct timespec end);

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
    
    if (argc < 6) {
        printf("correct usage: %s, [body count], [iterations], [config-name], " 
               "[simulation-name], [theta], [false: no output, true: output]\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int body_count = atoi(argv[1]);
    if (body_count < 1) {
        printf("body count must be at least 1\n");
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

    /*openCL setup*/
    cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("src/kernels/barnes_hut.ocl", ctx, d);
    
    cl_int err;
    cl_kernel reduce_min_k = clCreateKernel(prog, "reduce_min", &err);
    ocl_check(err, "clCreateKernel failed on reduce_min");

    cl_kernel reduce_max_k = clCreateKernel(prog, "reduce_max", &err);
    ocl_check(err, "clCreateKernel failed on reduce_max");

    cl_kernel reset_init_tree_k = clCreateKernel(prog, "reset_init_tree", &err );
    ocl_check(err, "clCreateKernel failed on reset_init_tree");

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
             reduction_min_event[(iterations + 1) * reduction_count],
             reduction_max_event[(iterations + 1) * reduction_count],
             reset_init_tree_event[iterations + 1],
             build_tree_map_event[(iterations + 1) * TREE_MAPPED_BUFFERS],
             build_tree_unmap_event[(iterations + 1) * TREE_MAPPED_BUFFERS],
             summarize_tree_event[(iterations + 1) * MAX_TREE_DEPTH];

    struct timespec build_tree_cpu_start_event[iterations + 1],
                    build_tree_cpu_end_event[iterations + 1];

    if (wants_output) {
        char outputs_path_name[PATH_MAX + 1] = OUTPUTS_PATH;
        strcat(outputs_path_name, sim_name);
        mkdir(outputs_path_name, S_IRWXU);
    }

    char outputs_path_name[PATH_MAX + 1] = BENCHMARKS_PATH;
    strcat(outputs_path_name, sim_name);
    mkdir(outputs_path_name, S_IRWXU);


    /*actual sim*/
    cl_mem input_buffer = body_pos_mem;
    cl_mem output_buffer = reduction_buffer1;
    int remaining = body_count;

    /*(leapfrog halfstep) finding both minimum and maximum x and y 
      in the particle system to find the bounding box coordinates*/
    for (int r = 0; r < reduction_count; r++) {
        reduction_min_event[r + iterations * reduction_count] = reduction_run(
            que,
            reduce_min_k,
            input_buffer,
            output_buffer,
            bounding_box_mem,
            (int)(r == (reduction_count - 1)),
            remaining
        );
        clWaitForEvents(1, &reduction_min_event[r + iterations * reduction_count]);
        remaining = (remaining + 1) / 2;

        input_buffer = output_buffer;
        if (output_buffer == reduction_buffer1) {
            output_buffer = reduction_buffer2;
        } else {
            output_buffer = reduction_buffer1;
        }
    }

    input_buffer = body_pos_mem;
    output_buffer = reduction_buffer1;
    remaining = body_count;
    
    for (int r = 0; r < reduction_count; r++) {
        reduction_max_event[r + iterations * reduction_count] = reduction_run(
            que,
            reduce_max_k,
            input_buffer,
            output_buffer,
            bounding_box_mem,
            (int)(r == (reduction_count - 1)),
            remaining
        );
        clWaitForEvents(1, &reduction_max_event[r + iterations * reduction_count]);
        remaining = (remaining + 1) / 2;

        input_buffer = output_buffer;
        if (output_buffer == reduction_buffer1) {
            output_buffer = reduction_buffer2;
        } else {
            output_buffer = reduction_buffer1;
        }
    }

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

    /*serial tree building on CPU*/
    build_tree_cpu_run(
        que,
        body_pos_mem,
        cell_children_mem,
        cell_center_mem,
        cell_half_size_mem,
        body_count,
        max_cells,
        body_pos_buffer_size,
        children_buffer_size,
        cell_2dfloat_buffer_size,
        cell_1dfloat_buffer_size,
        &build_tree_map_event[iterations * TREE_MAPPED_BUFFERS],
        &build_tree_unmap_event[iterations * TREE_MAPPED_BUFFERS],
        &build_tree_cpu_start_event[iterations],
        &build_tree_cpu_end_event[iterations]
    );

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

            cl_mem input_buffer = body_pos_mem;
            cl_mem output_buffer = reduction_buffer1;
            remaining = body_count;
        
            for (int r = 0; r < reduction_count; r++) {
                reduction_min_event[r + i * reduction_count] = reduction_run(
                    que,
                    reduce_min_k,
                    input_buffer,
                    output_buffer,
                    bounding_box_mem,
                    (int)(r == (reduction_count - 1)),
                    remaining
                );
                clWaitForEvents(1, &reduction_min_event[r + i * reduction_count]);
                remaining = (remaining + 1) / 2;
            
                input_buffer = output_buffer;
                if (output_buffer == reduction_buffer1) {
                    output_buffer = reduction_buffer2;
                } else {
                    output_buffer = reduction_buffer1;
                }
            }
            
            input_buffer = body_pos_mem;
            output_buffer = reduction_buffer1;
            remaining = body_count;
            for (int r = 0; r < reduction_count; r++) {
                reduction_max_event[r + i * reduction_count] = reduction_run(
                    que,
                    reduce_max_k,
                    input_buffer,
                    output_buffer,
                    bounding_box_mem,
                    (int)(r == (reduction_count - 1)),
                    remaining
                );
                clWaitForEvents(1, &reduction_max_event[r + i * reduction_count]);
                remaining = (remaining + 1) / 2;
                input_buffer = output_buffer;
                if (output_buffer == reduction_buffer1) {
                    output_buffer = reduction_buffer2;
                } else {
                    output_buffer = reduction_buffer1;
                }
            }
        
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
        
            build_tree_cpu_run(
                que,
                body_pos_mem,
                cell_children_mem,
                cell_center_mem,
                cell_half_size_mem,
                body_count,
                max_cells,
                body_pos_buffer_size,
                children_buffer_size,
                cell_2dfloat_buffer_size,
                cell_1dfloat_buffer_size,
                &build_tree_map_event[i * TREE_MAPPED_BUFFERS],
                &build_tree_unmap_event[i * TREE_MAPPED_BUFFERS],
                &build_tree_cpu_start_event[i],
                &build_tree_cpu_end_event[i]
            );
        
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
        
            cl_mem input_buffer = body_pos_mem;
            cl_mem output_buffer = reduction_buffer1;
            remaining = body_count;
        
            for (int r = 0; r < reduction_count; r++) {
                reduction_min_event[r + i * reduction_count] = reduction_run(
                    que,
                    reduce_min_k,
                    input_buffer,
                    output_buffer,
                    bounding_box_mem,
                    (int)(r == (reduction_count - 1)),
                    remaining
                );
                clWaitForEvents(1, &reduction_min_event[r + i * reduction_count]);
                remaining = (remaining + 1) / 2;
            
                input_buffer = output_buffer;
                if (output_buffer == reduction_buffer1) {
                    output_buffer = reduction_buffer2;
                } else {
                    output_buffer = reduction_buffer1;
                }
            }
            
            input_buffer = body_pos_mem;
            output_buffer = reduction_buffer1;
            remaining = body_count;
            for (int r = 0; r < reduction_count; r++) {
                reduction_max_event[r + i * reduction_count] = reduction_run(
                    que,
                    reduce_max_k,
                    input_buffer,
                    output_buffer,
                    bounding_box_mem,
                    (int)(r == (reduction_count - 1)),
                    remaining
                );
                clWaitForEvents(1, &reduction_max_event[r + i * reduction_count]);
                remaining = (remaining + 1) / 2;
                input_buffer = output_buffer;
                if (output_buffer == reduction_buffer1) {
                    output_buffer = reduction_buffer2;
                } else {
                    output_buffer = reduction_buffer1;
                }
            }
        
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
        
            build_tree_cpu_run(
                que,
                body_pos_mem,
                cell_children_mem,
                cell_center_mem,
                cell_half_size_mem,
                body_count,
                max_cells,
                body_pos_buffer_size,
                children_buffer_size,
                cell_2dfloat_buffer_size,
                cell_1dfloat_buffer_size,
                &build_tree_map_event[i * TREE_MAPPED_BUFFERS],
                &build_tree_unmap_event[i * TREE_MAPPED_BUFFERS],
                &build_tree_cpu_start_event[i],
                &build_tree_cpu_end_event[i]
            );
        
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
           reduction_min_ms = 0,
           reduction_max_ms = 0,
           reset_init_tree_ms = 0,
           build_tree_map_ms = 0,
           build_tree_cpu_ms = 0,
           build_tree_unmap_ms = 0,
           summarize_tree_ms = 0;

    for (int i = 0; i < iterations; i++) {
        compute_acc_ms += runtime_ms(compute_acc_walk_event[i]);
        update_pos_ms += runtime_ms(update_pos_event[i]);
        update_vel_ms += runtime_ms(update_vel_event[i]);
        for (int j = 0; j < reduction_count; j++) {
            reduction_min_ms += runtime_ms(reduction_min_event[i * reduction_count + j]);
            reduction_max_ms += runtime_ms(reduction_max_event[i * reduction_count + j]);
        }
        reset_init_tree_ms += runtime_ms(reset_init_tree_event[i]);
        for (int j = 0; j < TREE_MAPPED_BUFFERS; j++) {
            build_tree_map_ms += runtime_ms(build_tree_map_event[i * TREE_MAPPED_BUFFERS + j]);
            build_tree_unmap_ms += runtime_ms(build_tree_unmap_event[i * TREE_MAPPED_BUFFERS + j]);
        }
        build_tree_cpu_ms += timespec_elapsed_ms(
            build_tree_cpu_start_event[i],
            build_tree_cpu_end_event[i]
        );
        for (int j = 0; j < MAX_TREE_DEPTH; j++) {
            summarize_tree_ms += runtime_ms(summarize_tree_event[i * MAX_TREE_DEPTH + j]);
        }
    }
    
    compute_acc_ms += runtime_ms(compute_acc_walk_event[iterations]);
    update_vel_ms += runtime_ms(update_vel_event[iterations]);
    for (int j = 0; j < reduction_count; j++) {
            reduction_min_ms += runtime_ms(reduction_min_event[iterations * reduction_count + j]);
            reduction_max_ms += runtime_ms(reduction_max_event[iterations * reduction_count + j]);
    }
    reset_init_tree_ms += runtime_ms(reset_init_tree_event[iterations]);
    for (int j = 0; j < TREE_MAPPED_BUFFERS; j++) {
        build_tree_map_ms += runtime_ms(build_tree_map_event[iterations * TREE_MAPPED_BUFFERS + j]);
        build_tree_unmap_ms += runtime_ms(build_tree_unmap_event[iterations * TREE_MAPPED_BUFFERS + j]);
    }
    build_tree_cpu_ms += timespec_elapsed_ms(
        build_tree_cpu_start_event[iterations],
        build_tree_cpu_end_event[iterations]
    );
    for (int j = 0; j < MAX_TREE_DEPTH; j++) {
        summarize_tree_ms += runtime_ms(summarize_tree_event[iterations * MAX_TREE_DEPTH + j]);
    }


    double build_tree_ms = build_tree_map_ms + build_tree_cpu_ms + build_tree_unmap_ms;

    printf("TIMES:\n\nreduction_min: %fms\nreduction_max: %fms\nreset_init: %fms\n"
           "build + map + unmap: %fms\nsummarize: %fms\ncompute_acc: %fms\n"
           "update_vel: %fms\nupdate_pos: %fms\n",
           reduction_min_ms, reduction_max_ms, reset_init_tree_ms,
           build_tree_ms,summarize_tree_ms, compute_acc_ms, update_vel_ms, 
           update_pos_ms
        );

    write_bh_stats_on_disk(reduction_min_ms, reduction_max_ms, reset_init_tree_ms, build_tree_ms,summarize_tree_ms, 
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
    
    clReleaseKernel(reduce_min_k);
    clReleaseKernel(reduce_max_k);
    clReleaseKernel(reset_init_tree_k);
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
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_mass),
        &body_mass
    );
    ocl_check(err, "clSetKernelArg body_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_acc),
        &body_acc
    );
    ocl_check(err, "clSetKernelArg body_acc");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_mass),
        &cell_mass
    );
    ocl_check(err, "clSetKernelArg cell_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center),
        &cell_center
    );
    ocl_check(err, "clSetKernelArg cell_center");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_half_size),
        &cell_half_size
    );
    ocl_check(err, "clSetKernelArg cell_half_size");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center_of_mass),
        &cell_center_of_mass
    );
    ocl_check(err, "clSetKernelArg cell_center_of_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_children),
        &cell_children
    );
    ocl_check(err, "clSetKernelArg cell_children");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(theta_squared),
        &theta_squared
    );
    ocl_check(err, "clSetKernelArg theta_squared");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_count),
        &body_count
    );
    ocl_check(err, "clSetKernelArg body_count");
    arg++;

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
    arg++;
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_vel), &body_vel);
    ocl_check(err,"clSetKernelArg body_vel");
    arg++;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(delta_time), &delta_time);
    ocl_check(err,"clSetKernelArg update_pos delta_time");
    arg++;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg body_count");
    arg++;

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
    arg++;
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(body_acc), 
        &body_acc
    );
    ocl_check(err,"clSetKernelArg body_mass");
    arg++;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(delta_time), 
        &delta_time
    );
    ocl_check(err,"clSetKernelArg update_pos delta_time");
    arg++;

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


cl_event reduction_run(
    cl_command_queue que,
    cl_kernel k,
    cl_mem red_bufA,
    cl_mem red_bufB,
    cl_mem bounding_box,
    unsigned int is_last,
    unsigned int remaining_body_count
) {
    cl_event event;
    cl_int err;

    unsigned int output_count = (remaining_body_count + 1) / 2;
    cl_uint arg = 0;

    const size_t gws[1] = { round_mul_up(output_count, 32) };
    
    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(red_bufA), 
        &red_bufA
    );
    ocl_check(err, "clSetKernelArg red_bufA");
    arg++;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(red_bufB), 
        &red_bufB
    );
    ocl_check(err, "clSetKernelArg red_bufB");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(bounding_box),
        &bounding_box
    );
    ocl_check(err, "clSetKernelArg bounding_box");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(is_last),
        &is_last
    );
    ocl_check(err, "clSetKernelArg is_last");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(remaining_body_count),
        &remaining_body_count
    );
    ocl_check(err, "clSetKernelArg remaining_body_count");


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
    ocl_check(err, "clEnqueueNDRangeKernel reduce_min");

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
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center),
        &cell_center
    );
    ocl_check(err, "clSetKernelArg cell_center");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_half_size),
        &cell_half_size
    );
    ocl_check(err, "clSetKernelArg cell_half_size");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_mass),
        &cell_mass
    );
    ocl_check(err, "clSetKernelArg cell_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(bounding_box),
        &bounding_box
    );
    ocl_check(err, "clSetKernelArg bounding_box");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(max_cells),
        &max_cells
    );
    ocl_check(err, "clSetKernelArg max_cells");
    arg++;


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


int get_quadrant(cl_float2 pos, cl_float2 center) {
    int right = pos.x >= center.x;
    int top = pos.y >= center.y;
    return (top << 1) | right;
}


cl_float2 get_new_center(cl_float2 parent_center, cl_float parent_half_size, unsigned int quadrant) {
    cl_float2 new_center = parent_center;
    parent_half_size = parent_half_size * 0.5f;

    switch(quadrant) {
        case 0: new_center = (cl_float2) {new_center.x - parent_half_size, new_center.y - parent_half_size};
            break;
        case 1: new_center = (cl_float2) {new_center.x + parent_half_size, new_center.y - parent_half_size};
            break;
        case 2: new_center = (cl_float2) {new_center.x - parent_half_size, new_center.y + parent_half_size};
            break;
        case 3: new_center = (cl_float2) {new_center.x + parent_half_size, new_center.y + parent_half_size};
            break;
        default: new_center = (cl_float2) {new_center.x + parent_half_size, new_center.y + parent_half_size};
    }

    return new_center;
}


int cell_to_idx(int cell_id, int body_count) {
    return cell_id + body_count;
}


int idx_to_cell(int cell_id, int body_count) {
    return cell_id - body_count;
}


int idx_is_cell(int ref, int body_count) {
    return ref >= body_count;
}


void serial_build_tree_cpu(
    const cl_float2 *body_pos,
    cl_int *cell_children,
    cl_float2 *cell_center,
    cl_float *cell_half_size,
    unsigned int body_count,
    unsigned int max_cells
) {

    unsigned int allocated_cells = 1;

    for (int id = 0; id < body_count; id++) {
        cl_float2 this_pos = body_pos[id];
        int cell = 0;
        bool inserted = 0;

        for (int depth = 0; depth < MAX_TREE_DEPTH; depth++) {
            if (inserted) {
                break;
            }

            cl_float2 center = cell_center[cell];
            cl_float half_size = cell_half_size[cell];
            int quadrant = get_quadrant(this_pos, center);
            int child_idx = cell * CHILDREN + quadrant;
            int child = cell_children[child_idx];

            if (child == NULL_NODE) {
                cell_children[child_idx] = id;
                inserted = true;
                break;
            }

            if (idx_is_cell(child, body_count)) {
                cell = idx_to_cell(child, body_count);
                continue;
            }

            int old_body = child;
            cl_float2 old_pos = body_pos[old_body];

            int top_cell = -1;
            int parent_cell = -1;
            int parent_quadrant = -1;
            cl_float2 parent_center = center;
            cl_float parent_half_size = half_size;

            for (int split_depth = depth;
                split_depth < MAX_TREE_DEPTH;
                split_depth++
            ) {
                if (allocated_cells >= max_cells) {
                    return;
                }

                int new_cell = allocated_cells;
                allocated_cells++;
                cl_float new_half_size = parent_half_size * 0.5f;
                cl_float2 new_center = get_new_center(parent_center, parent_half_size, quadrant);
                cell_center[new_cell] = new_center;
                cell_half_size[new_cell] = new_half_size;

                for (int child_index = 0; child_index < CHILDREN; child_index++) {
                    cell_children[new_cell * CHILDREN + child_index] = NULL_NODE;
                }

                if (top_cell < 0) {
                    top_cell = new_cell;
                }

                if (parent_cell >= 0) {
                    cell_children[parent_cell * CHILDREN + parent_quadrant] = cell_to_idx(new_cell, body_count);
                }

                int old_quadrant = get_quadrant(old_pos, new_center);
                int new_quadrant = get_quadrant(this_pos, new_center);

                if (old_quadrant != new_quadrant) {
                    cell_children[new_cell * CHILDREN + old_quadrant] = old_body;
                    cell_children[new_cell * CHILDREN + new_quadrant] = id;
                    cell_children[child_idx] = cell_to_idx(top_cell, body_count);

                    inserted = true;
                    break;
                }

                parent_cell = new_cell;
                parent_quadrant = old_quadrant;
                parent_center = new_center;
                parent_half_size = new_half_size;
                quadrant = old_quadrant;
            }
        }
    }
}


double timespec_elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}


void build_tree_cpu_run(
    cl_command_queue que,
    cl_mem body_pos,
    cl_mem cell_children,
    cl_mem cell_center,
    cl_mem cell_half_size,
    unsigned int body_count,
    unsigned int max_cells,
    size_t body_pos_buffer_size,
    size_t children_buffer_size,
    size_t cell_2dfloat_buffer_size,
    size_t cell_1dfloat_buffer_size,
    cl_event *map_events,
    cl_event *unmap_events,
    struct timespec *cpu_start_event,
    struct timespec *cpu_end_event
) {

    cl_int err;

    cl_float2 *body_pos_map = clEnqueueMapBuffer(
        que,
        body_pos,
        CL_TRUE,
        CL_MAP_READ,
        0,
        body_pos_buffer_size,
        0,
        NULL,
        &map_events[0],
        &err
    );
    ocl_check(err, "clEnqueueMapBuffer body_pos build_tree_cpu");

    cl_int *cell_children_map = clEnqueueMapBuffer(
        que,
        cell_children,
        CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE,
        0,
        children_buffer_size,
        0,
        NULL,
        &map_events[1],
        &err
    );
    ocl_check(err, "clEnqueueMapBuffer cell_children build_tree_cpu");

    cl_float2 *cell_center_map = clEnqueueMapBuffer(
        que,
        cell_center,
        CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE,
        0,
        cell_2dfloat_buffer_size,
        0,
        NULL,
        &map_events[2],
        &err
    );
    ocl_check(err, "clEnqueueMapBuffer cell_center build_tree_cpu");

    cl_float *cell_half_size_map = clEnqueueMapBuffer(
        que,
        cell_half_size,
        CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE,
        0,
        cell_1dfloat_buffer_size,
        0,
        NULL,
        &map_events[3],
        &err
    );
    ocl_check(err, "clEnqueueMapBuffer cell_half_size build_tree_cpu");

    clock_gettime(CLOCK_MONOTONIC, cpu_start_event);

    serial_build_tree_cpu(
        body_pos_map,
        cell_children_map,
        cell_center_map,
        cell_half_size_map,
        body_count,
        max_cells
    );

    clock_gettime(CLOCK_MONOTONIC, cpu_end_event);

    err = clEnqueueUnmapMemObject(
        que,
        body_pos,
        body_pos_map,
        0,
        NULL,
        &unmap_events[0]
    );
    ocl_check(err, "clEnqueueUnmapMemObject body_pos build_tree_cpu");

    err = clEnqueueUnmapMemObject(
        que,
        cell_children,
        cell_children_map,
        0,
        NULL,
        &unmap_events[1]
    );
    ocl_check(err, "clEnqueueUnmapMemObject cell_children build_tree_cpu");

    err = clEnqueueUnmapMemObject(
        que,
        cell_center,
        cell_center_map,
        0,
        NULL,
        &unmap_events[2]
    );
    ocl_check(err, "clEnqueueUnmapMemObject cell_center build_tree_cpu");

    err = clEnqueueUnmapMemObject(
        que,
        cell_half_size,
        cell_half_size_map,
        0,
        NULL,
        &unmap_events[3]
    );
    ocl_check(err, "clEnqueueUnmapMemObject cell_half_size build_tree_cpu");

    clWaitForEvents(TREE_MAPPED_BUFFERS, unmap_events);
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
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_mass),
        &body_mass
    );
    ocl_check(err, "clSetKernelArg body_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_mass),
        &cell_mass
    );
    ocl_check(err, "clSetKernelArg cell_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_center_of_mass),
        &cell_center_of_mass
    );
    ocl_check(err, "clSetKernelArg cell_center_of_mass");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(cell_children),
        &cell_children
    );
    ocl_check(err, "clSetKernelArg cell_children");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(body_count),
        &body_count
    );
    ocl_check(err, "clSetKernelArg body_count");
    arg++;

    err = clSetKernelArg(
        k,
        arg,
        sizeof(max_cells),
        &max_cells
    );
    ocl_check(err, "clSetKernelArg max_cells");
    arg++;

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
