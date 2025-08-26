def get_dynamics(env, args, logger, writer):

    if args.algo_name in [
        "cac-approx",
        "c3m-approx",
        "ppo-approx",
        "sd-lqr-approx",
        "lqr-approx",
    ]:
        from policy.layers.dynamic_networks import DynamicLearner
        from trainer.offline_trainer import DynamicsTrainer

        print("[INFO] Learning a dynamics approximator.")
        # learn dynamics
        Dynamic_func = DynamicLearner(
            x_dim=env.num_dim_x,
            action_dim=args.action_dim,
            hidden_dim=args.DynamicLearner_dim,
            Dynamic_lr=args.Dynamic_lr,
            nupdates=args.dynamics_epochs,
            device=args.device,
        )
        Dynamic_trainer = DynamicsTrainer(
            env=env,
            Dynamic_func=Dynamic_func,
            logger=logger,
            writer=writer,
            buffer_size=args.dynamics_buffer_size,
            epochs=args.dynamics_epochs,
        )
        Dynamic_trainer.train()
        # env.replace_dynamics(Dynamic_func)

        init_epochs = args.dynamics_epochs
    else:
        init_epochs = 0
        Dynamic_func = env.get_f_and_B

    return Dynamic_func, init_epochs
