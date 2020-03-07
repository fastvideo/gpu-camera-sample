varying vec2 texcoord;
attribute highp vec4 vertexPosAttr;
attribute highp vec4 texPosAttr;
void main(void)
{
    gl_Position = vertexPosAttr;
    texcoord = texPosAttr.xy;
}
