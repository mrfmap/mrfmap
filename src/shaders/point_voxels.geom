#version 410
#define WIREFRAME 1
layout(points) in;
#ifndef WIREFRAME
layout(triangle_strip, max_vertices = 24) out;
#else
layout(line_strip, max_vertices = 24) out;
#endif

flat in lowp vec3 gColor[];
// flat in int gEnabledFaces[];

flat out lowp vec3 fColor;

uniform vec3 voxSize;// = {1.0f, 1.0f, 1.0f};
uniform mat4 mvp;
uniform bool draw_bounds = false;

void AddQuad(vec4 center, vec4 dy, vec4 dx) {
    fColor = gColor[0];
    gl_Position = center + (dx - dy);
    EmitVertex();

    fColor = gColor[0];
    gl_Position = center + (-dx - dy);
    EmitVertex();

    fColor = gColor[0];
    gl_Position = center + (dx + dy);
    EmitVertex();

    fColor = gColor[0];
    gl_Position = center + (-dx + dy);
    EmitVertex();

    EndPrimitive();
}

void AddFace(vec4 center, vec4 dy, vec4 dx) {
    fColor = gColor[0];
    gl_Position = center + (dx - dy);
    EmitVertex();

    fColor = gColor[0];
    gl_Position = center + (-dx - dy);
    EmitVertex();

    fColor = gColor[0];
    gl_Position = center + (-dx + dy);
    EmitVertex();

    fColor = gColor[0];
    gl_Position = center + (dx + dy);
    EmitVertex();

    EndPrimitive();
}

bool IsCulled(vec4 normal) {
    return normal.z > 0;
}

void main() {
    vec4 center = gl_in[0].gl_Position;
    
    vec4 dx = mvp[0] / 2.0f * voxSize[0];
    vec4 dy = mvp[1] / 2.0f * voxSize[1];
    vec4 dz = mvp[2] / 2.0f * voxSize[2];
#ifndef WIREFRAME
    // if ((gEnabledFaces[0] & 0x01) != 0 && !IsCulled(dx))
        AddQuad(center + dx, dy, dz);
        // if ((gEnabledFaces[0] & 0x02) != 0 && !IsCulled(-dx))
        AddQuad(center - dx, dz, dy);
    // if ((gEnabledFaces[0] & 0x04) != 0 && !IsCulled(dy))
        AddQuad(center + dy, dz, dx);
    // if ((gEnabledFaces[0] & 0x08) != 0 && !IsCulled(-dy))
        AddQuad(center - dy, dx, dz);
    // if ((gEnabledFaces[0] & 0x10) != 0 && !IsCulled(dz))
        AddQuad(center + dz, dx, dy);
    // if ((gEnabledFaces[0] & 0x20) != 0 && !IsCulled(-dz))
        AddQuad(center - dz, dy, dx);
#else
        AddFace(center + dx, dy, dz);
        AddFace(center - dx, dz, dy);
        AddFace(center + dy, dz, dx);
        AddFace(center - dy, dx, dz);
        AddFace(center + dz, dx, dy);
        AddFace(center - dz, dy, dx);
#endif
}