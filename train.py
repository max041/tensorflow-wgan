import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
from data import DATASETS, DATASET_LENGTH_GETTERS


def _sample_z(sample_size, z_size):
    return np.random.uniform(
        -1., 1., size=[sample_size, z_size]
    ).astype(np.float32)


def train(model, config, session=None):
    # define a session if needed.
    session = session or tf.Session()

    # define summaries.
    summary_writer = tf.summary.FileWriter(config.log_dir, session.graph)
    image_summary = tf.summary.image(
        'generated images', model.G, max_outputs=8
    )
    loss_summaries = tf.summary.merge([
        tf.summary.scalar('wasserstein distance', -model.c_loss),
        tf.summary.scalar('generator loss', model.g_loss),
    ])

    # define optimizers.
    C_traner = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1
    )
    G_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
    )

    # define parameter update tasks
    c_grads = C_traner.compute_gradients(model.c_loss, var_list=model.c_vars)
    g_grads = G_trainer.compute_gradients(model.g_loss, var_list=model.g_vars)
    update_C = C_traner.apply_gradients(c_grads)
    update_G = G_trainer.apply_gradients(g_grads)
    clip_C = [
        v.assign(tf.clip_by_value(v, -config.clip_size, config.clip_size))
        for v in model.c_vars
    ]

    if config.execution_graph_dump_to:
        import os
        tf.summary.FileWriter(os.path.join(os.getcwd(), config.execution_graph_dump_to), tf.get_default_graph())
        exit(0)

    # main training session context
    with session:
        if config.resume:
            epoch_start = utils.load_checkpoint(session, model, config) + 1
        else:
            epoch_start = 1
            session.run(tf.global_variables_initializer())

        for epoch in range(epoch_start, config.epochs+1):
            dataset = DATASETS[config.dataset](config.batch_size)
            dataset_length = DATASET_LENGTH_GETTERS[config.dataset]()
            dataset_stream = tqdm(enumerate(dataset, 1))

            for batch_index, xs in dataset_stream:
                # where are we?
                iteration = (epoch-1)*(dataset_length // config.batch_size) + batch_index

                # place more weight on ciritic in the begining of the training.
                critic_update_ratio = (
                    30 if (batch_index < 25 or batch_index % 500 == 0) else
                    config.critic_update_ratio
                )

                # train the critic against the current generator and the data.
                for _ in range(critic_update_ratio):
                    zs = _sample_z(config.batch_size, model.z_size)

                    if config.critic_dump_to:
                        import os
                        from tensorflow.python import debug as tf_debug
                        run_options = tf.RunOptions()
                        tf_debug.watch_graph(
                            run_options,
                            session.graph,
                            debug_urls=['file://' + os.path.join(os.getcwd(), config.critic_dump_to)]
                        )
                        _, c_loss = session.run(
                            [update_C, model.c_loss],
                            feed_dict={
                                model.z_in: zs,
                                model.image_in: xs
                            },
                            options=run_options
                        )
                        # session.run(clip_C, options=run_options)
                        exit(0)

                    _, c_loss = session.run(
                        [update_C, model.c_loss],
                        feed_dict={
                            model.z_in: zs,
                            model.image_in: xs
                        }
                    )
                    session.run(clip_C)

                # train the generator against the current critic.
                zs = _sample_z(config.batch_size, model.z_size)

                if config.generator_dump_to:
                    import os
                    from tensorflow.python import debug as tf_debug
                    run_options = tf.RunOptions()
                    tf_debug.watch_graph(
                        run_options,
                        session.graph,
                        debug_urls=['file://' + os.path.join(os.getcwd(), config.generator_dump_to)]
                    )
                    _, g_loss = session.run(
                        [update_G, model.g_loss],
                        feed_dict={model.z_in: zs},
                        options=run_options
                    )
                    exit(0)

                _, g_loss = session.run(
                    [update_G, model.g_loss],
                    feed_dict={model.z_in: zs}
                )

                # display current training process status
                dataset_stream.set_description((
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'g loss: {g_loss:.3f} | '
                    'w distance: {w_dist:.3f}'
                ).format(
                    epoch=epoch,
                    epochs=config.epochs,
                    trained=batch_index*config.batch_size,
                    total=dataset_length,
                    progress=(
                        100.
                        * batch_index
                        * config.batch_size
                        / dataset_length
                    ),
                    g_loss=g_loss,
                    w_dist=-c_loss,
                ))

                # log the generated samples
                if iteration % config.image_log_interval == 0:
                    zs = _sample_z(config.sample_size, model.z_size)
                    summary_writer.add_summary(session.run(
                        image_summary, feed_dict={
                            model.z_in: zs
                        }
                    ), iteration)

                # log the losses
                if iteration % config.loss_log_interval == 0:
                    zs = _sample_z(config.batch_size, model.z_size)
                    summary_writer.add_summary(session.run(
                        loss_summaries, feed_dict={
                            model.z_in: zs,
                            model.image_in: xs
                        }
                    ), iteration)

            # save the model at the every end of the epochs.
            utils.save_checkpoint(session, model, epoch, config)


def train_original(model, config, session=None):
    # define a session if needed.
    session = session or tf.Session()

    # define summaries.
    summary_writer = tf.summary.FileWriter(config.log_dir, session.graph)
    image_summary = tf.summary.image(
        'generated images', model.G, max_outputs=8
    )
    loss_summaries = tf.summary.merge([
        tf.summary.scalar('wasserstein distance', -model.c_loss),
        tf.summary.scalar('generator loss', model.g_loss),
    ])

    # define optimizers.
    C_trainer = tf.train.RMSPropOptimizer(
        learning_rate=config.learning_rate
    )
    G_trainer = tf.train.RMSPropOptimizer(
        learning_rate=config.learning_rate
    )

    # define parameter update tasks
    c_grads = C_trainer.compute_gradients(model.c_loss, var_list=model.c_vars)
    g_grads = G_trainer.compute_gradients(model.g_loss, var_list=model.g_vars)
    update_C = C_trainer.apply_gradients(c_grads)
    update_G = G_trainer.apply_gradients(g_grads)
    clip_C = [
        v.assign(tf.clip_by_value(v, -config.clip_size, config.clip_size))
        for v in model.c_vars
    ]

    if config.execution_graph_dump_to:
        import os
        tf.summary.FileWriter(os.path.join(os.getcwd(), config.execution_graph_dump_to), tf.get_default_graph())
        exit(0)

    # main training session context
    with session:
        if config.resume:
            epoch_start = utils.load_checkpoint(session, model, config) + 1
        else:
            epoch_start = 1
            session.run(tf.global_variables_initializer())

        step_counter = 0
        for epoch in range(epoch_start, config.epochs+1):
            dataset = DATASETS[config.dataset](config.batch_size)
            dataset_length = DATASET_LENGTH_GETTERS[config.dataset]()
            dataset_stream = tqdm(dataset_length//config.batch_size)

            try:
                # while theta has not converged do
                while True:
                    # for t=0,...,n_critic do
                    for _ in range(config.critic_update_ratio):
                        # Sample {x^(i)}[i=1,m]~Pr a batch from the real data.
                        dataset_stream.update()
                        xs = next(dataset)
                        # Sample {z^(i)}[i=1,m]~p(z) a batch of prior samples.
                        zs = _sample_z(config.batch_size, model.z_size)

                        if config.critic_dump_to:
                            import os
                            from tensorflow.python import debug as tf_debug
                            run_options = tf.RunOptions()
                            tf_debug.watch_graph(
                                run_options,
                                session.graph,
                                debug_urls=['file://' + os.path.join(os.getcwd(), config.critic_dump_to)]
                            )
                            _, c_loss = session.run(
                                [update_C, model.c_loss],
                                feed_dict={
                                    model.z_in: zs,
                                    model.image_in: xs
                                },
                                options=run_options
                            )
                            exit(0)

                        # g_w <- grad_w[mean(f_w(x^(i))) - mean(f_w(g_theta(z^(i))))]
                        # w <- w + alpha * RMSProp(w, g_w)
                        _, c_loss = session.run(
                            [update_C, model.c_loss],
                            feed_dict={
                                model.z_in: zs,
                                model.image_in: xs
                            }
                        )
                        # w <- clip(w, -c, c)
                        session.run(clip_C)

                    # end for
                    # Sample {z^(i)}[i=1,m]~p(z) a batch of prior samples.
                    zs = _sample_z(config.batch_size, model.z_size)

                    if config.generator_dump_to:
                        import os
                        from tensorflow.python import debug as tf_debug
                        run_options = tf.RunOptions()
                        tf_debug.watch_graph(
                            run_options,
                            session.graph,
                            debug_urls=['file://' + os.path.join(os.getcwd(), config.generator_dump_to)]
                        )
                        _, g_loss = session.run(
                            [update_G, model.g_loss],
                            feed_dict={model.z_in: zs},
                            options=run_options
                        )
                        exit(0)

                    # g_theta <- -grad_theta[mean(f_w(g_theta(z^(i))))]
                    # theta <- theta - alpha * RMSProp(theta, g_theta)
                    _, g_loss = session.run(
                        [update_G, model.g_loss],
                        feed_dict={model.z_in: zs}
                    )



                    # display current training process status
                    step_counter += 1
                    dataset_stream.set_description((
                        'epoch: {epoch}/{epochs} | '
                        'progress: [{trained}/{total}] ({progress:.0f}%) | '
                        'g loss: {g_loss:.3f} | '
                        'w distance: {w_dist:.3f}'
                    ).format(
                        epoch=epoch,
                        epochs=config.epochs,
                        trained=(config.batch_size * step_counter) % dataset_length,
                        total=dataset_length,
                        progress=(
                            100. * (config.batch_size * step_counter) % dataset_length / dataset_length
                        ),
                        g_loss=g_loss,
                        w_dist=-c_loss,
                    ))
                    # log the generated samples
                    if step_counter % config.image_log_interval == 0:
                        zs = _sample_z(config.sample_size, model.z_size)
                        summary_writer.add_summary(session.run(
                            image_summary, feed_dict={
                                model.z_in: zs
                            }
                        ), step_counter)

                    # log the losses
                    if step_counter % config.loss_log_interval == 0:
                        zs = _sample_z(config.batch_size, model.z_size)
                        summary_writer.add_summary(session.run(
                            loss_summaries, feed_dict={
                                model.z_in: zs,
                                model.image_in: xs
                            }
                        ), step_counter)
                # end while
            except:
                pass

            # save the model at the every end of the epochs.
            utils.save_checkpoint(session, model, epoch, config)