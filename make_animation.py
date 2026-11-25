import numpy as np
import plotly.graph_objects as go


def build_vortex_time_animation(psi, grid, azim_deg=45.0, elev_deg=25.0, zoom=1.6, phase_t=0.0, time_label=""):
    # Эволюция волны во времени psi(r, t) = psi(r) * exp(-i * phase_t)
    # Визуализируем огибающую + бегущую фазовую волну Re(psi * exp(-i * phase_t)) при неподвижной камере
    psi_t = psi * np.exp(-1j * float(phase_t))
    amp = np.abs(psi_t)
    wave_field = (amp ** 2) * (0.65 + 0.35 * np.cos(np.angle(psi_t)))
    u_max = float(np.max(wave_field))
    if u_max <= 0:
        u_max = 1.0

    X_flat = grid["X"].flatten()
    Y_flat = grid["Y"].flatten()
    Z_flat = grid["Z"].flatten()
    V_flat = wave_field.flatten()

    # Перевод углов камеры и зума в координаты Plotly scene.camera.eye
    r_cam = 2.6 / max(float(zoom), 0.2)
    az0 = np.radians(float(azim_deg))
    el0 = np.radians(float(elev_deg))

    eye_x0 = r_cam * np.cos(el0) * np.cos(az0)
    eye_y0 = r_cam * np.cos(el0) * np.sin(az0)
    eye_z0 = r_cam * np.sin(el0)

    # Базовый кадр (фиксированный ракурс и приближение)
    base_trace = go.Isosurface(
        x=X_flat,
        y=Y_flat,
        z=Z_flat,
        value=V_flat,
        isomin=0.25 * u_max,
        isomax=0.85 * u_max,
        surface_count=3,
        colorscale="Turbo",
        opacity=0.70,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    # Генерация кадров плавного облёта вокруг зафиксированного ракурса и зума
    frames = []
    angles = [0.0]
#    angles = np.linspace(0.0, 2.0 * np.pi, int(n_frames), endpoint=False)
    

    for idx, d_az in enumerate(angles):
        az = az0 + d_az
        ex = r_cam * np.cos(el0) * np.cos(az)
        ey = r_cam * np.cos(el0) * np.sin(az)
        ez = eye_z0
        frames.append(
            go.Frame(
                name=f"f_{idx}",
                layout=dict(
                    scene=dict(
                        camera=dict(
                            eye=dict(x=float(ex), y=float(ey), z=float(ez)),
                            up=dict(x=0.0, y=0.0, z=1.0),
                        )
                    )
                ),
            )
        )

    fig = go.Figure(data=[base_trace], frames=frames)
    fig.update_layout(
 #       title=f"3D-анимация узла (ракурс: {azim_deg:.0f}°, наклон: {elev_deg:.0f}°, зум: {zoom:.1f}×)",

        title=f"3D-анимация узла | {time_label} (ракурс: {azim_deg:.0f}°, наклон: {elev_deg:.0f}°, зум: {zoom:.1f}×)",

        height=560,
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(x=float(eye_x0), y=float(eye_y0), z=float(eye_z0)),
