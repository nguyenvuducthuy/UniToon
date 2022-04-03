#ifndef UNITOON_FUNCTIONS_INCLUDED
#define UNITOON_FUNCTIONS_INCLUDED

inline half invlerp(const half start, const half end, const half t)
{
    return (t - start) / (end - start);
}

inline half remap(const half v, const half fromMin, const half fromMax, const half toMin, const half toMax)
{
    return toMin + (v - fromMin) * (toMax - toMin) / (fromMax - fromMin);
}

half3 shift(half3 color, half3 shift)
{
    half VSU = shift.z * shift.y * cos(shift.x * 6.28318512);
    half VSW = shift.z * shift.y * sin(shift.x * 6.28318512);
        
    return half3(
        (0.299 * shift.z + 0.701 * VSU + 0.168 * VSW) * color.r + (0.587 * shift.z - 0.587 * VSU + 0.330 * VSW) * color.g + (0.114 * shift.z - 0.114 * VSU - 0.497 * VSW) * color.b,
        (0.299 * shift.z - 0.299 * VSU - 0.328 * VSW) * color.r + (0.587 * shift.z + 0.413 * VSU + 0.035 * VSW) * color.g + (0.114 * shift.z - 0.114 * VSU + 0.292 * VSW) * color.b,
        (0.299 * shift.z - 0.300 * VSU + 1.25 * VSW)  * color.r + (0.587 * shift.z - 0.588 * VSU - 1.05 * VSW)  * color.g + (0.114 * shift.z + 0.886 * VSU - .203 * VSW) * color.b
    );
}

inline half maxcolor(half3 color)
{
    return max(color.r, max(color.g, color.b));
}

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"

SamplerState my_linear_clamp_sampler;

float sampleSceneDepth(float2 uv)
{
    float sceneDepth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, my_linear_clamp_sampler, uv);
    return Linear01Depth(sceneDepth, _ZBufferParams) * _ProjectionParams.z;
}

float SoftOutline(float2 uv, half width, half strength, half power)
{
    float sceneDepth = sampleSceneDepth(uv);
    float w = width / max(sceneDepth * 1.0, 1.0);
    width = (w + 0.5) * 0.5;
    float2 delta = (1.0 / _ScreenParams.xy) * width;

    const int SAMPLE = 8;
    float depthes[SAMPLE];
    depthes[0] = sampleSceneDepth(uv + float2(-delta.x, -delta.y));
    depthes[1] = sampleSceneDepth(uv + float2(-delta.x,  0.0)    );
    depthes[2] = sampleSceneDepth(uv + float2(-delta.x,  delta.y));
    depthes[3] = sampleSceneDepth(uv + float2(0.0,      -delta.y));
    depthes[4] = sampleSceneDepth(uv + float2(0.0,       delta.y));
    depthes[5] = sampleSceneDepth(uv + float2(delta.x,  -delta.y));
    depthes[6] = sampleSceneDepth(uv + float2(delta.x,   0.0)    );
    depthes[7] = sampleSceneDepth(uv + float2(delta.x,   delta.y));

    float coeff[SAMPLE] = {0.7071, 1.0, 0.7071, 1.0, 1.0, 0.7071, 1.0, 0.7071};

    float depthValue = 0;
    float str = pow(20.0, strength * 10.0) * 0.5;
    float smoothness = 1.0 / remap(power, 0.0, 1.0, 0.01, 0.3);
    [unroll]
    for (int j = 0; j < SAMPLE; j++)
    {
        float sub = abs(depthes[j] - sceneDepth);
        sub = pow(sub, smoothness);
        sub *= str * coeff[j];
        depthValue += sub;
    }

    half outlineRate = saturate(depthValue);

    return outlineRate;
}

// global addition transform
float3 _pos;
float3 _rot;
float3 _scl;
#include "Packages/com.sparx.unicin/Core/Shader/common.hlsl"
VertexPositionInputs GetVertexPositionInputsThuy(float3 positionOS)
{
    // global addition transform
    float3x3 rot = Euler3x3(_rot);
    positionOS.xyz  = mul(rot,positionOS.xyz);
    positionOS.xyz *= _scl.x <= 0 ? 1 : _scl;
    positionOS.xyz += _pos;

    VertexPositionInputs input;
    input.positionWS = TransformObjectToWorld(positionOS);
    input.positionVS = TransformWorldToView(input.positionWS);
    input.positionCS = TransformWorldToHClip(input.positionWS);

    float4 ndc = input.positionCS * 0.5f;
    input.positionNDC.xy = float2(ndc.x, ndc.y * _ProjectionParams.x) + ndc.w;
    input.positionNDC.zw = input.positionCS.zw;

    return input;
}

VertexNormalInputs GetVertexNormalInputsThuy(float3 normalOS)
{
    // global addition transform
    float3x3 rot = Euler3x3(_rot);
    normalOS.xyz  = mul(rot,normalOS.xyz);
    normalOS.xyz *= _scl.x <= 0 ? 1 : _scl;
    normalOS.xyz += _pos;

    VertexNormalInputs tbn;
    tbn.tangentWS = real3(1.0, 0.0, 0.0);
    tbn.bitangentWS = real3(0.0, 1.0, 0.0);
    tbn.normalWS = TransformObjectToWorldNormal(normalOS);
    return tbn;
}

VertexNormalInputs GetVertexNormalInputsThuy(float3 normalOS, float4 tangentOS)
{
    // global addition transform
    float3x3 rot = Euler3x3(_rot);
    normalOS.xyz  = mul(rot,normalOS.xyz);
    normalOS.xyz *= _scl.x <= 0 ? 1 : _scl;
    normalOS.xyz += _pos;

    tangentOS.xyz  = mul(rot,tangentOS.xyz);
    tangentOS.xyz *= _scl.x <= 0 ? 1 : _scl;
    tangentOS.xyz += _pos;

    VertexNormalInputs tbn;

    // mikkts space compliant. only normalize when extracting normal at frag.
    real sign = real(tangentOS.w) * GetOddNegativeScale();
    tbn.normalWS = TransformObjectToWorldNormal(normalOS);
    tbn.tangentWS = real3(TransformObjectToWorldDir(tangentOS.xyz));
    tbn.bitangentWS = real3(cross(tbn.normalWS, float3(tbn.tangentWS))) * sign;
    return tbn;
}

#endif
