kernel void saxpy(
	float a,
	global float* x,
	global float* y)
{
	int gid = get_global_id(0);

	y[gid] = a * x[gid] + y[gid];
}