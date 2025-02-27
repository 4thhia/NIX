import os
import matplotlib.pyplot as plt
import imageio
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np

from nix.toy.common.losses import create_loss_grad_fn, loss_fn_surface


def plot_lambda(cfg, history):
    steps = range(len(history["lmbs"]))
    lmb_values = history["lmbs"]

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lmb_values, label="lmb", color='b', marker='o')

    # title and label
    plt.title(r'$\lambda$ over Training Steps')
    plt.xlabel('Steps')
    plt.ylabel(r'$\lambda$')
    plt.legend()
    output_dir = f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{cfg.experiment_name}.png"))


def plot_utils(cfg, history):

    steps = range(len(history["lmbs"]))
    weights = history["weights"]
    params_total_opt = history["params_total_opt"]
    lmb_gain = history["lmb_gain"]
    main_loss = history["main_loss"]
    gamma = history["gamma"]
    lmb_values = history["lmbs"]

    NUM_COLUMNS = 2
    NUM_ROWS = 3
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLUMNS, figsize=(NUM_COLUMNS*8, NUM_ROWS*5))

    axes[0, 0].plot(steps, weights, color="dodgerblue", linewidth=2)
    axes[0, 1].plot(steps, params_total_opt, color="dodgerblue", linewidth=2)
    axes[0, 1].axhline(y=cfg.main.mux[0], color="red", linestyle="--", linewidth=1.5)
    axes[1, 0].plot(steps, lmb_gain, color="dodgerblue", linewidth=2)
    axes[1, 1].plot(steps, gamma, color="dodgerblue", linewidth=2)
    axes[2, 0].plot(steps, main_loss, color="dodgerblue", linewidth=2)
    axes[2, 0].axhline(y=cfg.algorithm.target_loss, color="red", linestyle="--", linewidth=1.5)
    axes[2, 1].plot(steps, lmb_values, color="dodgerblue", linewidth=2)

    axes[0, 0].set_title("weights", fontsize=18)
    axes[0, 1].set_title("total_opt", fontsize=18)
    axes[1, 0].set_title("constraint", fontsize=18)
    axes[1, 1].set_title("gamma", fontsize=18)
    axes[2, 0].set_title("main_loss", fontsize=18)
    axes[2, 1].set_title("lmb_values", fontsize=18)

    plt.tight_layout()
    output_dir = f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{cfg.experiment_name}.png"))


def plot2D(cfg, history):
    main_loss_fns, _ = create_loss_grad_fn(cfg.main.mux, cfg.main.muy, cfg.main.stdx, cfg.main.stdy, cfg.main.rho, cfg.main.flat)
    aux_loss_fns, _ = create_loss_grad_fn(cfg.aux.mux, cfg.aux.muy, cfg.aux.stdx, cfg.aux.stdy, cfg.aux.rho, cfg.aux.flat)

    def loss_fn(x, weights):
        weights = np.array(weights)[:, np.newaxis, np.newaxis]
        main_loss = np.sum(np.array([fn(x) for fn in main_loss_fns]))
        weighted_aux_loss = np.sum(weights * np.array([fn(x) for fn in aux_loss_fns]), axis=0)
        return main_loss + weighted_aux_loss

    x = np.linspace(-cfg.plot.lim, cfg.plot.lim, 200)
    y = np.linspace(-cfg.plot.lim, cfg.plot.lim, 200)
    X, Y = np.meshgrid(x, y)

    # initialize
    fig, ax = plt.subplots()
    gif_images = []

    num_points = len(history["params"])

    for i in range(num_points):
        ax.clear()  # Reset plot region

        #
        point = history["params"][i]
        ax.scatter(point[0], point[1], color='k', s=20, label='Point' if i == 0 else "")

        main_grad = history["main_grads"][i]
        aux_grad = history["weighted_aux_grads"][i]

        # Grad Norm
        main_grad_scaled = np.array(main_grad)
        aux_grad_scaled = np.array(aux_grad)

        # Grad arrow
        ax.quiver(point[0], point[1], main_grad_scaled[0], main_grad_scaled[1], angles='xy', scale_units='xy', scale=1, width=0.003, color='r', label='Main Gradient' if i == 0 else "")
        ax.quiver(point[0], point[1], aux_grad_scaled[0], aux_grad_scaled[1], angles='xy', scale_units='xy', scale=1, width=0.003, color='b', label='Weighted Aux Gradient' if i == 0 else "")

        # Contour
        Z_main = np.sum(np.array([fn([X, Y]) for fn in main_loss_fns]), axis=0)
        Z_total = loss_fn([X, Y], history["weights"][i])
        ax.contour(X, Y, Z_main, levels=10, cmap='Pastel1')
        ax.contour(X, Y, Z_total, levels=10, cmap='viridis')

        for j in range(len(cfg.main.mux)):
            plt.scatter(cfg.main.mux[j], cfg.main.muy[j], color='r', marker='x', s=100)
        for k in range(len(cfg.aux.mux)):
            plt.scatter(cfg.aux.mux[k], cfg.aux.muy[k], color='b', marker='x', s=100)


        ax.set_xlim(-cfg.plot.lim, cfg.plot.lim)
        ax.set_ylim(-cfg.plot.lim, cfg.plot.lim)
        ax.set_title(fr'step:{i} $\lambda$: {history["lmbs"][i]}')

        plt.draw()
        plt.grid(True)
        plt.pause(0.1)
        img_path = f"/tmp/temp_{i}.png"
        plt.savefig(img_path)
        gif_images.append(imageio.imread(img_path))

    # Save gif
    output_dir = f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}"
    os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(os.path.join(output_dir, f"{cfg.experiment_name}.gif"), gif_images, duration=0.1)
    plt.close(fig)


def plot3D(cfg, history):
    main_loss_fns, _ = create_loss_grad_fn(cfg.main.mux, cfg.main.muy, cfg.main.stdx, cfg.main.stdy, cfg.main.rho, cfg.main.flat)
    aux_loss_fns, _ = create_loss_grad_fn(cfg.aux.mux, cfg.aux.muy, cfg.aux.stdx, cfg.aux.stdy, cfg.aux.rho, cfg.aux.flat)

    def loss_fn(x, weights):
        main_loss = np.sum(np.array([fn(x) for fn in main_loss_fns]))
        weighted_aux_loss = np.sum(weights * np.array([fn(x) for fn in aux_loss_fns]))
        return main_loss + weighted_aux_loss

    main_loss_fn_surfaces = loss_fn_surface(cfg.main.mux, cfg.main.muy, cfg.main.stdx, cfg.main.stdy, cfg.main.rho, cfg.main.flat, cfg.plot.lim)
    xs, ys, _ = main_loss_fn_surfaces[0]
    zs = sum([zs for _, _, zs in main_loss_fn_surfaces], np.zeros_like(main_loss_fn_surfaces[0][2]))
    main_loss_fn_surface = (xs, ys, zs)

    aux_loss_fn_surfaces = loss_fn_surface(cfg.aux.mux, cfg.aux.muy, cfg.aux.stdx, cfg.aux.stdy, cfg.aux.rho, cfg.aux.flat, cfg.plot.lim)

    k = 0
    # Set initial frame
    initial_xyz = main_loss_fn_surface
    for w, surface in zip(history["weights"][k], aux_loss_fn_surfaces):
        initial_xyz = (
            initial_xyz[0],
            initial_xyz[1],
            initial_xyz[2] + w * surface[2]  # z方向を更新
        )

    fig = go.Figure(data=[
        go.Surface(
            x=initial_xyz[0],
            y=initial_xyz[1],
            z=initial_xyz[2],
            colorscale="viridis",
            opacity=0.8,
            showscale=False,
            contours_z=dict(
                show=True,
                usecolormap=True,
                project_z=True,
                size=0.01,
            )
        ),
        go.Surface(
            x=main_loss_fn_surface[0],
            y=main_loss_fn_surface[1],
            z=np.full_like(main_loss_fn_surface[2], np.min(main_loss_fn_surface[2]) * 1.05),
            surfacecolor=main_loss_fn_surface[2],
            colorscale="earth",
            showscale=False,
            opacity=0.5,
        )
    ])

    x, y = np.array(history["params"])[:, 0], np.array(history["params"])[:, 1]

    frames = []
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "step:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 100, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    for k in range(np.array(history["params"]).shape[0]):
        current_xyz = main_loss_fn_surface
        for w, surface in zip(history["weights"][k], aux_loss_fn_surfaces):
            current_xyz = (
                current_xyz[0],
                current_xyz[1],
                current_xyz[2] + w * surface[2]
            )

        frames.append(go.Frame(
            data=[
                go.Surface(
                    x=current_xyz[0],
                    y=current_xyz[1],
                    z=current_xyz[2],
                    colorscale="viridis",
                    opacity=0.8,
                    showscale=False,
                    contours_z=dict(
                        show=True,
                        usecolormap=True,
                        project_z=True,
                        size=0.01,
                    )
                ),
                go.Surface(
                    x=main_loss_fn_surface[0],
                    y=main_loss_fn_surface[1],
                    z=np.full_like(main_loss_fn_surface[2], np.min([np.min(main_loss_fn_surface[2]), np.min(current_xyz[2])]) * 1.05),
                    surfacecolor=main_loss_fn_surface[2],
                    colorscale="earth",
                    showscale=False,
                    opacity=0.6,
                ),
                go.Scatter3d(
                    x=[x[k]],
                    y=[y[k]],
                    z=[loss_fn([x[k], y[k]], history["weights"][k])],
                    mode="markers",
                    marker=dict(color="red", size=6, opacity=1),
                )
            ],
            name=f"step{k}"
        ))
        slider_step = {"args": [
            [f"step{k}"],
            {"frame": {"duration": 100, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 100}}
        ],
            "label": k,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig.update(frames=frames)
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    redraw=True,
                                    duration=100
                                ),
                                fromcurrent=True,
                                transition=dict(
                                    duration=100,
                                    easing="quadratic-in-out"
                                )
                            )
                        ]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[
                            [None],
                            dict(
                                frame=dict(
                                    redraw=False,
                                    duration=0
                                ),
                                mode="immediate",
                                fromcurrent=True,
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ],
                direction="left",
                pad=dict(
                    r=10,
                    t=87
                ),
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ],
        sliders=[sliders_dict],
        scene=dict(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        )
    )

    pio.write_html(fig, file=f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.experiment_name}.html", auto_open=False)
