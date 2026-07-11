#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  float2 resolution;
  float2 fineCenter;
  float2 coarseCenter;
  float2 baseCenterUV;
  float2 baseUVPerM;
  float2 siteOrigin;
  float4 camPos;
  float4 camFwd;
  float4 camRight;
  float4 camUp;
  float4 sunDir;
  float time;
  float world;
  float fineWorld;
  float hScale;
  float apron;
  float coarseWorld;
  float hMax;
  float hasBase;
  float baseTexelM;
  float hasAlbedo;
  float albedoRGB;
  float streamAlb;
  float albedoGain;
  float padR;
  float synthDetail;
  float fov;
  float ambient;
  float shadowStrength;
  float synthAmp;
  float fineAmp;
  float grainAmp;
  float fogAmp;
  int baseCount;
  int baseTarget;
  int debug;
  int ss;
  int _pad0;
  int _pad1;
  float4 bases[8];
};

vertex float4 moon_vertex(uint vid [[vertex_id]]) {
  float2 p = float2((vid << 1) & 2, vid & 2);
  return float4(p * 2.0 - 1.0, 0.0, 1.0);
}

float2 baseUV(float2 world, constant Uniforms& u){
  return u.baseCenterUV + (world - float2(u.world * 0.5)) * u.baseUVPerM;
}
float worldScale(constant Uniforms& u){ return u.world / 4096.0; }
float tfar(constant Uniforms& u){ return u.world * 3.0; }
float lodAt(float t, float texelW, constant Uniforms& u){
  float footprint = t * (2.0*u.fov / u.resolution.y);
  return max(0.0, log2(footprint / texelW));
}
float hash(float2 p){ return fract(sin(dot(p, float2(127.1,311.7))) * 43758.5453); }
float vnoise(float2 p){
  float2 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  float a = hash(i), b = hash(i+float2(1.0,0.0));
  float c = hash(i+float2(0.0,1.0)), d = hash(i+float2(1.0,1.0));
  return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}
float synthRelief(float2 wM, constant Uniforms& u){
  return (0.60*(vnoise(wM*0.0033)-0.5) + 0.28*(vnoise(wM*0.009)-0.5) + 0.14*(vnoise(wM*0.026)-0.5)) * (60.0 * u.synthAmp);
}
float synthFade(float t, constant Uniforms& u){
  float fpM = t * (2.0*u.fov / u.resolution.y);
  float f = clamp(1.0 - fpM / 45.0, 0.0, 1.0);
  return f*f;
}
float fineHeight(float2 wM, float fpM, constant Uniforms& u){
  float s = 0.55*(vnoise(wM*0.017)-0.5)*clamp(1.0 - fpM/12.0, 0.0, 1.0)
          + 0.30*(vnoise(wM*0.045)-0.5)*clamp(1.0 - fpM/4.0,  0.0, 1.0)
          + 0.16*(vnoise(wM*0.120)-0.5)*clamp(1.0 - fpM/1.3,  0.0, 1.0);
  return s * (55.0 * u.fineAmp);
}
float terrain(float2 world, float t, constant Uniforms& u, sampler s,
              texture2d<float> uHeightTex, texture2d<float> uFineTex, texture2d<float> uBaseTex){
  float2 cuv = (world - u.coarseCenter) / u.coarseWorld + 0.5;
  float2 ccl = clamp(cuv, 0.0, 1.0);
  float h = uHeightTex.sample(s, ccl, level(lodAt(t, u.coarseWorld / float(uHeightTex.get_width()), u))).r * u.hScale;
  float outM = length((cuv - ccl) * u.coarseWorld);
  if (outM > 0.0) {
    if (u.hasBase > 0.5) {
      float hb = uBaseTex.sample(s, clamp(baseUV(world, u), 0.0, 1.0), level(lodAt(t, u.baseTexelM, u))).r * u.hScale;
      h = mix(h, hb, smoothstep(0.0, u.coarseWorld * 0.04, outM));
    } else {
      h = mix(h, u.apron, smoothstep(0.0, u.coarseWorld * 0.15, outM));
    }
  }
  if (u.fineWorld > 0.0) {
    float2 fuv = (world - u.fineCenter) / u.fineWorld + 0.5;
    if (fuv.x > 0.0 && fuv.x < 1.0 && fuv.y > 0.0 && fuv.y < 1.0) {
      float hf = uFineTex.sample(s, fuv, level(lodAt(t, u.fineWorld / float(uFineTex.get_width()), u))).r * u.hScale;
      float edge = min(min(fuv.x, 1.0-fuv.x), min(fuv.y, 1.0-fuv.y));
      h = mix(h, hf, smoothstep(0.0, 0.04, edge));
    }
  }
  return h;
}
float marchDetail(float2 world, float t, constant Uniforms& u){
  float f = synthFade(t, u);
  return f > 0.0 ? synthRelief(world, u) * f : 0.0;
}
float4 bsplW(float t){ float t2 = t*t, t3 = t2*t;
  return float4((1.0-t)*(1.0-t)*(1.0-t), 3.0*t3 - 6.0*t2 + 4.0,
              -3.0*t3 + 3.0*t2 + 3.0*t + 1.0, t3) / 6.0; }
float4 bsplD(float t){ float t2 = t*t;
  return float4(-(1.0-t)*(1.0-t), 3.0*t2 - 4.0*t, -3.0*t2 + 2.0*t + 1.0, t2) / 2.0; }
float2 gradLevel(texture2d<float> tex, sampler s, float2 uv, float lod){
  float res = float(tex.get_width()) / exp2(lod);
  float2 st = uv * res - 0.5;
  float2 i = floor(st), f = st - i;
  float4 wx = bsplW(f.x), wy = bsplW(f.y);
  float4 dx = bsplD(f.x), dy = bsplD(f.y);
  float gx = 0.0, gy = 0.0;
  for (int r = 0; r < 4; r++){
    float rowV = 0.0, rowD = 0.0;
    for (int c = 0; c < 4; c++){
      float p = tex.sample(s, (i + float2(float(c) - 1.0, float(r) - 1.0) + 0.5) / res, level(lod)).r;
      rowD += dx[c] * p; rowV += wx[c] * p;
    }
    gx += wy[r] * rowD; gy += dy[r] * rowV;
  }
  return float2(gx, gy) * res;
}
float2 gradC1(texture2d<float> tex, sampler s, float2 uv, float lod){
  float l0 = floor(lod);
  float2 g = gradLevel(tex, s, uv, l0);
  float fl = lod - l0;
  if (fl > 0.0) g = mix(g, gradLevel(tex, s, uv, l0 + 1.0), fl);
  return g;
}
float2 terrainGrad(float2 world, float t, constant Uniforms& u, sampler s,
                   texture2d<float> uHeightTex, texture2d<float> uFineTex, texture2d<float> uBaseTex){
  float2 cuv0 = (world - u.coarseCenter) / u.coarseWorld + 0.5;
  float2 cuv = clamp(cuv0, 0.0, 1.0);
  float clod = lodAt(t, u.coarseWorld / float(uHeightTex.get_width()), u);
  float2 g = gradC1(uHeightTex, s, cuv, clod) * (u.hScale / u.coarseWorld);
  float outM = length((cuv0 - cuv) * u.coarseWorld);
  if (outM > 0.0) {
    if (u.hasBase > 0.5) {
      float2 gb = gradC1(uBaseTex, s, clamp(baseUV(world, u), 0.0, 1.0), lodAt(t, u.baseTexelM, u)) * (u.hScale * u.baseUVPerM);
      g = mix(g, gb, smoothstep(0.0, u.coarseWorld * 0.04, outM));
    } else {
      g *= 1.0 - smoothstep(0.0, u.coarseWorld * 0.15, outM);
    }
  }
  if (u.fineWorld > 0.0) {
    float2 fuv = (world - u.fineCenter) / u.fineWorld + 0.5;
    if (fuv.x > 0.0 && fuv.x < 1.0 && fuv.y > 0.0 && fuv.y < 1.0) {
      float flod = lodAt(t, u.fineWorld / float(uFineTex.get_width()), u);
      float2 gf = gradC1(uFineTex, s, fuv, flod) * (u.hScale / u.fineWorld);
      float edge = min(min(fuv.x, 1.0-fuv.x), min(fuv.y, 1.0-fuv.y));
      g = mix(g, gf, smoothstep(0.0, 0.04, edge));
    }
  }
  return g;
}
float3 terrainNormal(float2 world, float t, constant Uniforms& u, sampler s,
                     texture2d<float> uHeightTex, texture2d<float> uFineTex, texture2d<float> uBaseTex){
  float2 g = terrainGrad(world, t, u, s, uHeightTex, uFineTex, uBaseTex);
  return normalize(float3(-g.x, 1.0, -g.y));
}
float3 terrainNormalFine(float2 world, float t, constant Uniforms& u, sampler s,
                         texture2d<float> uHeightTex, texture2d<float> uFineTex, texture2d<float> uBaseTex){
  float e = 3.0;
  float fpM = t * (2.0*u.fov / u.resolution.y);
  float2 g = terrainGrad(world, t, u, s, uHeightTex, uFineTex, uBaseTex);
  float dL = marchDetail(world-float2(e,0.0),t,u) + fineHeight(world-float2(e,0.0),fpM,u);
  float dR = marchDetail(world+float2(e,0.0),t,u) + fineHeight(world+float2(e,0.0),fpM,u);
  float dD = marchDetail(world-float2(0.0,e),t,u) + fineHeight(world-float2(0.0,e),fpM,u);
  float dU = marchDetail(world+float2(0.0,e),t,u) + fineHeight(world+float2(0.0,e),fpM,u);
  g += float2(dR - dL, dU - dD) / (2.0*e);
  return normalize(float3(-g.x, 1.0, -g.y));
}
float fbm(float2 p){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 3; i++){ s += a*vnoise(p); p *= 2.03; a *= 0.5; }
  return s;
}
float3 shade(float2 fc, constant Uniforms& u, sampler s,
             texture2d<float> uHeightTex, texture2d<float> uFineTex, texture2d<float> uAlbedoTex,
             texture2d<float> uBaseTex, texture2d<float> uFineAlbTex, texture2d<float> uCoarseAlbTex){
  float3 uCamPos = u.camPos.xyz, uSunDir = u.sunDir.xyz;
  float2 ndc = (fc / u.resolution) * 2.0 - 1.0;
  ndc.x *= u.resolution.x / u.resolution.y;
  float3 dir = normalize(u.camFwd.xyz + ndc.x*u.fov*u.camRight.xyz + ndc.y*u.fov*u.camUp.xyz);
  float3 col;
  float WS = worldScale(u);
  float far = tfar(u);
  float t = 1.0, tPrev = 1.0;
  bool  hit = false;
  for (int i = 0; i < 768; i++){
    float3 p = uCamPos + dir * t;
    if (p.y > u.hMax && dir.y >= 0.0) break;
    float h = terrain(p.xz, t, u, s, uHeightTex, uFineTex, uBaseTex);
    if (u.synthDetail > 0.5 && (p.y - h) < 50.0 * u.synthAmp) h += marchDetail(p.xz, t, u);
    if (p.y < h){ hit = true; break; }
    tPrev = t;
    t += max(2.0, min((p.y - h) * 0.45, t * 0.02));
    if (t > far) break;
  }
  if (hit){
    float tHit = t;
    for (int j = 0; j < 8; j++){
      float tm = 0.5*(tHit+tPrev);
      float3 p = uCamPos + dir*tm;
      float hm = terrain(p.xz, tm, u, s, uHeightTex, uFineTex, uBaseTex);
      if (u.synthDetail > 0.5) hm += marchDetail(p.xz, tm, u);
      if (p.y < hm) tHit = tm; else tPrev = tm;
    }
    float3 p = uCamPos + dir*tHit;
    float3 n = (u.synthDetail > 0.5) ? terrainNormalFine(p.xz, tHit, u, s, uHeightTex, uFineTex, uBaseTex)
                                     : terrainNormal(p.xz, tHit, u, s, uHeightTex, uFineTex, uBaseTex);
    float diff = max(dot(n, uSunDir), 0.0);
    if (u.debug == 1) return float3(diff);
    if (u.debug == 2) return float3(clamp(p.y / u.hMax, 0.0, 1.0));
    if (u.debug == 3) return n * 0.5 + 0.5;
    if (u.debug == 4) {
      float dmin = 1e9;
      for (int bi = 0; bi < 8; bi++) {
        if (bi >= u.baseCount) break;
        dmin = min(dmin, length(p.xz - u.bases[bi].xz));
      }
      return float3(1.0 - clamp(dmin / 8000.0, 0.0, 1.0),
                    fract(dmin / 500.0) * 0.5,
                    u.baseCount > 0 ? 0.25 : 0.0);
    }
    float sh = 1.0;
    if (diff > 0.0 && u.shadowStrength > 0.001) {
      float tt = 14.0 * WS;
      for (int k = 0; k < 32; k++){
        float3 sp = p + uSunDir * tt;
        if (sp.y > u.hMax) break;
        float clr = sp.y - terrain(sp.xz, tHit, u, s, uHeightTex, uFineTex, uBaseTex);
        if (clr < 0.0){ sh = 0.0; break; }
        sh = min(sh, 14.0 * clr / tt);
        tt += max(20.0 * WS, tt * 0.12);
      }
      sh = mix(1.0 - 0.70 * u.shadowStrength, 1.0, clamp(sh, 0.0, 1.0));
    }
    float3 albedo;
    float aFadeM = 0.0, aFadeSpan = 1.0;
    if (u.streamAlb > 0.5) {
      float2 cuv0 = (p.xz - u.coarseCenter) / u.coarseWorld + 0.5;
      float2 ccl = clamp(cuv0, 0.0, 1.0);
      float3 a = uCoarseAlbTex.sample(s, ccl, level(lodAt(tHit, u.coarseWorld / float(uCoarseAlbTex.get_width()), u))).rgb;
      if (u.fineWorld > 0.0) {
        float2 fuv = (p.xz - u.fineCenter) / u.fineWorld + 0.5;
        if (fuv.x > 0.0 && fuv.x < 1.0 && fuv.y > 0.0 && fuv.y < 1.0) {
          float3 af = uFineAlbTex.sample(s, fuv, level(lodAt(tHit, u.fineWorld / float(uFineAlbTex.get_width()), u))).rgb;
          float edge = min(min(fuv.x, 1.0-fuv.x), min(fuv.y, 1.0-fuv.y));
          a = mix(a, af, smoothstep(0.0, 0.04, edge));
        }
      }
      albedo = a * u.albedoGain;
      aFadeM = length((cuv0 - ccl) * u.coarseWorld);
      aFadeSpan = u.coarseWorld * 0.10;
    } else if (u.hasAlbedo > 0.5) {
      float2 auv0 = (p.xz - u.siteOrigin) / u.world;
      float2 auv = clamp(auv0, 0.0, 1.0);
      aFadeM = length((auv - auv0) * u.world);
      aFadeSpan = u.world * 0.10;
      float aLod = lodAt(tHit, u.world / float(uAlbedoTex.get_width()), u);
      if (u.albedoRGB > 0.5) {
        albedo = uAlbedoTex.sample(s, auv, level(aLod)).rgb * u.albedoGain;
      } else {
        float a = uAlbedoTex.sample(s, auv, level(aLod)).r;
        a = clamp((a - 0.37) * 1.8 + 0.37, 0.12, 0.70);
        albedo = a * float3(1.02, 1.00, 0.96) * u.albedoGain;
      }
    } else {
      float elev = clamp(p.y / u.hMax, 0.0, 1.0);
      albedo = mix(float3(0.28,0.27,0.26), float3(0.72,0.71,0.69), elev) * u.albedoGain;
    }
    float slope = clamp(1.0 - n.y, 0.0, 1.0);
    float aMod = 1.0 + 0.18 * u.grainAmp * slope;
    float fpM = tHit * (2.0*u.fov / u.resolution.y);
    float grain = clamp(1.0 - fpM / 25.0, 0.0, 1.0);
    if (grain > 0.0) aMod *= 1.0 + 0.16 * u.grainAmp * grain * (fbm(p.xz * 0.05) - 0.4);
    albedo *= aMod;
    if (aFadeM > 0.0) albedo = mix(albedo, float3(0.36), smoothstep(0.0, aFadeSpan, aFadeM));
    float padEmis = 0.0; float3 padCol = float3(0.0);
    for (int bi = 0; bi < 8; bi++) {
      if (bi >= u.baseCount) break;
      float d = length(p.xz - u.bases[bi].xz);
      float w = max(9.0, fpM * 1.6);
      if (d > u.padR + w * 4.0) continue;
      bool isT = (bi == u.baseTarget);
      float3 mark = isT ? float3(1.00, 0.72, 0.25) : float3(0.75, 0.78, 0.82);
      float ring = 1.0 - smoothstep(30.0, 30.0 + w, d);
      ring += 1.0 - smoothstep(w, w * 2.0, abs(d - 120.0));
      ring += 1.0 - smoothstep(w, w * 2.0, abs(d - 240.0));
      ring += 1.0 - smoothstep(w, w * 2.0, abs(d - 360.0));
      ring += 0.35 * (1.0 - smoothstep(w * 2.0, w * 4.0, abs(d - u.padR)));
      ring = clamp(ring, 0.0, 1.0) * clamp(1.0 - fpM / 60.0, 0.0, 1.0);
      if (ring > 0.0) {
        albedo = mix(albedo, mark, ring * 0.85);
        float pulse = isT ? (0.75 + 0.25 * sin(u.time * 2.2)) : 0.45;
        padEmis = max(padEmis, ring * pulse);
        padCol = mark;
      }
    }
    col = albedo * (u.ambient + diff * sh);
    col += padCol * padEmis * 0.35;
    float fog = smoothstep(far*0.25, far, tHit);
    col = mix(col, float3(0.05,0.05,0.07), clamp(fog * u.fogAmp, 0.0, 1.0));
  } else {
    col = float3(0.01,0.01,0.02);
    float2 sky = floor(dir.xy * 400.0);
    float sv = hash(sky);
    if (sv > 0.997 && dir.y > 0.0) col += float3(pow(sv,40.0)*3.0);
    col += float3(0.04,0.04,0.05) * smoothstep(0.15, 0.0, abs(dir.y));
  }
  if (u.debug >= 5 && u.debug <= 7) col = float3(0.0);
  for (int bi = 0; bi < 8; bi++) {
    if (bi >= u.baseCount) break;
    float2 dxz = u.bases[bi].xz - uCamPos.xz;
    float rr = dot(dir.xz, dir.xz);
    if (rr < 1e-6) continue;
    float tb = dot(dxz, dir.xz) / rr;
    if (u.debug != 5) {
      if (tb < 1.0 || (hit && tb > t * 1.03 + 80.0)) continue;
    }
    tb = max(tb, 1.0);
    float3 q = uCamPos + dir * tb;
    float yRel = (q.y - u.bases[bi].y) / 1400.0;
    if (u.debug != 5 && u.debug != 6) {
      if (yRel < 0.0 || yRel > 1.0) continue;
    }
    yRel = clamp(yRel, 0.0, 1.0);
    float hh = length(q.xz - u.bases[bi].xz);
    float sigma = 20.0 + tb * 0.004;
    float g = exp(-(hh * hh) / (2.0 * sigma * sigma));
    bool isT = (bi == u.baseTarget);
    float pulse = isT ? (0.65 + 0.35 * sin(u.time * 2.2)) : 0.40;
    float3 bc = isT ? float3(1.00, 0.70, 0.22) : float3(0.55, 0.75, 1.00);
    col += bc * (g * pulse * (1.0 - yRel * 0.75) * 0.8);
  }
  return col;
}
fragment float4 moon_fragment(float4 fragCoord [[position]],
                              constant Uniforms& u [[buffer(0)]],
                              texture2d<float> uHeightTex   [[texture(0)]],
                              texture2d<float> uFineTex     [[texture(1)]],
                              texture2d<float> uAlbedoTex   [[texture(2)]],
                              texture2d<float> uBaseTex     [[texture(3)]],
                              texture2d<float> uFineAlbTex  [[texture(4)]],
                              texture2d<float> uCoarseAlbTex[[texture(5)]],
                              sampler samp [[sampler(0)]]){
  // [[position]] is top-left origin; the GL shader assumed gl_FragCoord's
  // bottom-left. Flip Y back so the ray direction (ndc.y * uCamUp) matches.
  float2 fc0 = float2(fragCoord.x, u.resolution.y - fragCoord.y);
  int ss = clamp(u.ss, 1, 3);
  float3 acc = float3(0.0);
  for (int sy = 0; sy < ss; sy++)
  for (int sx = 0; sx < ss; sx++){
    float2 off = (float2(float(sx), float(sy)) + 0.5) / float(ss) - 0.5;
    acc += shade(fc0 + off, u, samp, uHeightTex, uFineTex, uAlbedoTex, uBaseTex, uFineAlbTex, uCoarseAlbTex);
  }
  return float4(pow(acc / float(ss*ss), float3(0.85)), 1.0);
}
