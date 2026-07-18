#include "./headers/ocl_boiler.h"
#include "./headers/sim-utils.h"
#include <sys/stat.h>
#include <math.h>
#include <limits.h>
#include <linux/limits.h>

#define DELTA_TIME 0.02f
#define CENTER_DISTANCE 10
#define GALAXIES_PATH "./galaxies/"
#define OUTPUTS_PATH "./outputs/"
#define SEED 42
#define MAX_TREE_DEPTH 32
#define NODE_PER_BODY 8
#define CHILDREN 4



cl_event update_force_run_k(
    cl_command_queue que, 
    cl_kernel k, 
    cl_mem body_pos, 
    cl_mem body_vel,
    cl_mem body_force,
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
    
    err = clSetKernelArg(k, arg_index, sizeof(body_vel), &body_vel);
    ocl_check(err,"clSetKernelArg body_vel");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_force), &body_force);
    ocl_check(err,"clSetKernelArg body_force");
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
    cl_mem body_pos, 
    cl_mem body_vel,
    cl_mem body_force,
    cl_mem body_mass,
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
    
    err = clSetKernelArg(k, arg_index, sizeof(body_force), &body_force);
    ocl_check(err,"clSetKernelArg body_mass");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(body_mass), &body_mass);
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


cl_event reduction_run_k(
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
    unsigned int arg = 0;

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

    unsigned int arg = 0;

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

    unsigned int arg = 0;

    const size_t gws[1] = { 1 };
    
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

    unsigned int arg = 0;

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


int main(int argc, char *argv[]) {
    
    if (argc < 6) {
        printf("correct usage: %s, [body count], [iterations], [config-name], [simulation-name], [theta]\n", argv[0]);
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
    float theta = atof(argv[5]);

    /*openCL shenanigans*/
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
    cl_kernel serial_build_tree_k = clCreateKernel(prog, "serial_build_tree", &err);
    ocl_check(err, "clCreateKernel failed on serial_build_tree");
    cl_kernel summarize_tree_k = clCreateKernel(prog, "summarize_tree", &err);
    ocl_check(err, "clCreateKernel failed on summarize_tree");


    cl_kernel update_pos_k = clCreateKernel(prog, "update_pos", &err);
    ocl_check(err, "clCreateKernel failed on update_pos");
        
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
    size_t body_force_buffer_size = sizeof(cl_float2) * body_count;
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
    while (fgets(line, sizeof(line), fp) != NULL) {
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

    cl_mem body_force_mem = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE,
        body_force_buffer_size,
        NULL,
        &err
    );
    ocl_check(err, "clCreateBuffer failed on body_force_mem");

    err = clEnqueueFillBuffer(
        que,
        body_force_mem,
        &(cl_int){0},
        sizeof(cl_float2),
        0,               
        body_force_buffer_size,
        0, 
        NULL, 
        NULL
    );
    ocl_check(err, "clEnqueueFillBuffer failed on body_force_mem");

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


    /*TODO
    CREARE TUTTI GLI EVENTI*/

    /*TODO
    MEZZO PASSO PER USARE IL LEAPFROG*/
    cl_event enqueue_map_or_read_buffer_event;
    char outputs_path_name[PATH_MAX + 1] = OUTPUTS_PATH;
    strcat(outputs_path_name, sim_name);
    mkdir(outputs_path_name, S_IRWXU);

    
    /*ITERAZIONI DELLA SIMULAZIONE*/
    for (int i = 0; i < iterations; i++) {

        cl_mem input_buffer = body_pos_mem;
        cl_mem output_buffer = reduction_buffer1;

        unsigned int remaining = body_count;

        /*TROVARE IL MINIMO DEL SISTEMA*/
        while (remaining > 1) {
            reduction_run_k(
                que,
                reduce_min_k,
                input_buffer,
                output_buffer,
                bounding_box_mem,
                (int)(remaining == 2),
                remaining
            );
            
            remaining = (remaining + 1) / 2;
            input_buffer = output_buffer;
        
            if (output_buffer == reduction_buffer1) {
                output_buffer = reduction_buffer2;
            } else {
                output_buffer = reduction_buffer1;
            }
        }

        remaining = body_count;
        
        /*TROVARE IL MASSIMO DEL SISTEMA*/
        input_buffer = body_pos_mem;
        output_buffer = reduction_buffer1;
        remaining = body_count;
        while (remaining > 1) {
            reduction_run_k(
                que,
                reduce_max_k,
                input_buffer,
                output_buffer,
                bounding_box_mem,
                (remaining == 2),
                remaining
            );
            
            remaining = (remaining + 1) / 2;
            input_buffer = output_buffer;
        
            if (output_buffer == reduction_buffer1) {
                output_buffer = reduction_buffer2;
            } else {
                output_buffer = reduction_buffer1;
            }
        }

        /* TODO: CREARE UNA MODALITÀ DEBUG PER STAMPARE TUTTI I PASSAGGI
        cl_float2 min, max;
        err = clEnqueueReadBuffer(
            que,
            bounding_box_mem,
            CL_TRUE,
            0,
            sizeof(min),
            &min,
            0,
            NULL,
            &enqueue_map_or_read_buffer_event
        );
        ocl_check(err, "clEnqueueReadBuffer failed");

        err = clEnqueueReadBuffer(
            que,
            bounding_box_mem,
            CL_TRUE,
            sizeof(min),
            sizeof(min),
            &max,
            0,
            NULL,
            &enqueue_map_or_read_buffer_event
        );

        ocl_check(err, "clEnqueueReadBuffer failed");

        printf("MINIMI TROVATI: %f, %f\nMASSIMI TROVATI: %f, %f\n", min.x, min.y, max.x, max.y);
        */

        cl_event reset_event = reset_init_tree_run(
            que,
            reset_init_tree_k,
            cell_children_mem,
            cell_center_mem,
            cell_half_size_mem,
            cell_mass_mem,
            bounding_box_mem,
            max_cells
        );

        cl_event build_event = build_tree_run(
            que,
            serial_build_tree_k,
            body_pos_mem,
            cell_children_mem,
            cell_center_mem,
            cell_half_size_mem,
            body_count,
            max_cells
        );

        /* TO DO: CREARE MODALITÀ DEBUG PER VALUTARE GLI OUTPUT
        cl_float2 *cell_center_cpu = (cl_float2*) malloc(cell_2dfloat_buffer_size);

        err = clEnqueueReadBuffer(
            que,
            cell_center_mem,
            CL_TRUE,
            0,
            cell_2dfloat_buffer_size,
            cell_center_cpu,
            0,
            NULL,
            &enqueue_map_or_read_buffer_event
        );
        ocl_check(err, "enqueueReadBuffer failed");

        cl_float *cell_halfsize_cpu = (cl_float*) malloc(cell_1dfloat_buffer_size);

        err = clEnqueueReadBuffer(
            que,
            cell_half_size_mem,
            CL_TRUE,
            0,
            cell_1dfloat_buffer_size,
            cell_halfsize_cpu,
            0,
            NULL,
            &enqueue_map_or_read_buffer_event
        );
        ocl_check(err, "enqueueReadBuffer failed");

        write_bounding_box_on_disk(max_cells, cell_center_cpu, cell_halfsize_cpu, sim_name, i);
        */
        /*for (int i = 0; i < max_cells; i++) {
            printf("%d\n", cell_children_cpu[i]);
        }*/

        #pragma unroll MAX_TREE_DEPTH
        for (int i = 0; i < MAX_TREE_DEPTH; i++) {
            summarize_tree_run(
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
        }

        update_pos_run_k(
            que,
            update_pos_k,
            body_pos_mem,
            body_vel_mem,
            body_count,
            DELTA_TIME
        );
        
        body_pos = clEnqueueMapBuffer(
            que, 
            body_pos_mem, 
            CL_TRUE, 
            CL_MAP_READ, 
            0, 
            body_pos_buffer_size, 
            0, 
            NULL, 
            &enqueue_map_or_read_buffer_event, 
            &err
        );
        ocl_check(err, "enqueueMapBuffer failed");
        
        /*err = clEnqueueReadBuffer(
            que,
            body_pos_mem,
            CL_TRUE,
            0,
            body_pos_buffer_size,
            body_pos,
            0,
            NULL,
            &enqueue_map_or_read_buffer_event
        );
        ocl_check(err, "enqueueReadBuffer failed");*/


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
    
    clFinish(que);
    /*
    for (int i = 0; i < 100; i++) {
        printf("%f %f\n", body_pos[i].x, body_pos[i].y);
    }*/

    clReleaseMemObject(body_pos_mem);
    clReleaseMemObject(body_vel_mem);
    clReleaseMemObject(body_mass_mem);
    clReleaseMemObject(body_force_mem);
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
    clReleaseKernel(serial_build_tree_k);
    clReleaseKernel(summarize_tree_k);
    clReleaseKernel(update_pos_k);

    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    
    return EXIT_SUCCESS;
}