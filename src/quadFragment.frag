uniform sampler2D tex;
varying mediump vec2 texcoord;
void main(void)
{
    gl_FragColor = texture2D(tex, texcoord);
}
