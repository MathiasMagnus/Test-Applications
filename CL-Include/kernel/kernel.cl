#include <kernel.h.cl>

__kernel void saxpy(real a,
                    __global real* x,
                    __global real* y)
{
	int gid = get_global_id(0);
	
	y[gid] = a * x[gid] + y[gid];
}
