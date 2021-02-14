StructuredBuffer<float> x : register(t0);    // SRV
RWStructuredBuffer<float> y : register(u0);  // UAV

cbuffer cbCS : register(b0)
{
    float a;
};

[numthreads(128, 1, 1)]
void CSMain(uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    y[DTid.x] = a * x[DTid.x] + y[DTid.x];
}