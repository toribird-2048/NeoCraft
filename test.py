import glfw
from OpenGL.GL import *
import numpy as np

def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(640, 480, "Simple Point", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # --- OpenGL 初期化 ---
    glClearColor(0.0, 0.0, 0.0, 1.0) # 画面クリア色を黒に設定

    # --- 頂点データ (点の座標) ---
    points = np.array([
        [0.0, 0.0, 0.0]  # 原点 (ウィンドウ中央)
    ], dtype=np.float32)

    # --- VBO (Vertex Buffer Object) 作成 ---
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)

    # --- VAO (Vertex Array Object) 作成 ---
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # 頂点属性の有効化と設定 (ここでは位置属性のみ)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None) # attribute index 0 を使用
    glEnableVertexAttribArray(0)

    # --- シェーダープログラム ---
    vertex_shader_source = """
        #version 460 core
        layout (location = 0) in vec3 aPos; // 頂点属性: 位置 (location=0)
        void main() {
            gl_Position = vec4(aPos, 1.0); // 頂点の位置をそのまま出力
            gl_PointSize = 10.0;          // 点のサイズを設定
        }
    """

    fragment_shader_source = """
        #version 460 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0); // フラグメントの色を白に設定 (RGBA)
        }
    """

    # バーテックスシェーダーのコンパイル
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    # エラーチェック (省略)

    # フラグメントシェーダーのコンパイル
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    # エラーチェック (省略)

    # シェーダープログラムの作成とリンク
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    # エラーチェック (省略)

    glDeleteShader(vertex_shader) # シェーダーオブジェクトは不要になったので削除
    glDeleteShader(fragment_shader) # シェーダーオブジェクトは不要になったので削除

    glUseProgram(shader_program) # シェーダープログラムの使用開始


    # --- 描画ループ ---
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT) # 画面をクリア

        glBindVertexArray(vao) # VAOをバインド
        glDrawArrays(GL_POINTS, 0, 1) # プリミティブタイプ: GL_POINTS, 描画開始頂点: 0, 描画頂点数: 1

        glfw.swap_buffers(window) # フロントバッファとバックバッファを入れ替え
        glfw.poll_events()      # イベント処理

    # --- リソース解放 ---
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(shader_program)

    glfw.terminate()

if __name__ == "__main__":
    main()