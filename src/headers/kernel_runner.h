/* PURE BARNES HUT V1 */
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
    ++arg;

    err = clSetKernelArg(
        k, 
        arg, 
        sizeof(red_bufB), 
        &red_bufB
    );
    ocl_check(err, "clSetKernelArg red_bufB");
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
        sizeof(is_last),
        &is_last
    );
    ocl_check(err, "clSetKernelArg is_last");
    ++arg;

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


/* PURE BARNES HUT V2*/


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
