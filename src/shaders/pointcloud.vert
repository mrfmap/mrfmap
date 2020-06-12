#version 410
layout (location = 0) in vec3 vPos; 

uniform mat4 mvp; 
const float z_max = 5.0f;

out vec3 fColor;

vec3 jet(float t)
{
  return clamp(vec3(1.5) - abs(4.0 * vec3(t) + vec3(-3, -2, -1)), vec3(0), vec3(1));
}

void main()
{
  gl_Position = mvp * vec4(vPos, 1.0); 
  gl_PointSize = 0.01f;
  fColor = jet(1.0f - vPos.z/z_max);
}