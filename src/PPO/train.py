# pip install gymnasium
import gymnasium as gym
import torch
import torch.optim as optim
from PPO import PPO

# 環境を定義
env = gym.make("Hopper-v5")

# GPUがあれば使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPOモデルの作成と最適化関数の作成
model: PPO = PPO().to(device=device)
optimizer = optim.Adam()

# ハイパーパラメータを定義
ppo_epochs = 4
rollout_steps = 2048
batch_size = 256
clip_eps = 0.2
gamma = 0.9
gae_lambda = 0.9
value_coef = 0.5
entropy_coef = 0.01  # log(std)の勾配に対応する
num_iterations = 1000

for iteration in range(num_iterations):
    # 1. rollout収集

    # 2. Tensor化

    # 3. GAE計算

    # 4. value target作成

    # 5. Advantage正規化

    # 6. 複数epochのPPO更新

    # 経験学習のためのbuffer
    states = []
    actions = []
    rewards = []
    dones = []
    old_log_probs = []
    values = []
    advantages = []

    '''
        ここからrolloutによるデータ集め
    '''
    # rollout_steps 回だけ環境と相互作用してデータを集めます。
    # エピソードを跨いでrollout_steps回のstepをこなす
    state, info = env.reset()
    state = torch.as_tensor(state,
                            dtype=torch.float32,
                            device=device)

    for _ in range(rollout_steps):
        with torch.no_grad():
            dist, value = model(state)

            # 行動分布から行動をサンプリングします。
            action = dist.sample()

            # π(a|s) = π(a1|s) * π(a2|s) * π(a3|s)
            # 各行動次元のlog probabilityを計算し、
            # 行動ベクトル全体のlog probabilityにまとめます。
            log_prob = dist.log_prob(action).sum(dim=-1)

            value = value.squeeze(-1)

        # envに渡すのはnumpyでcpuに渡す必要がある
        env_action = action.detach().cpu().numpy()
        next_state, reward, terminated, truncated, info = env.step(env_action)
        next_state = torch.as_tensor(next_state,
                                     dtype=torch.float32,
                                     device=device)

        # 環境が自然に終了した場合も、
        # TimeLimitなどで打ち切られた場合もリセット対象にします。
        done = terminated | truncated

        # 現在状態は後から書き換わる可能性に備えてコピーして保存します。
        states.append(state.detach().clone())

        # 実際に実行した行動を固定データとして保存します。
        actions.append(action.detach())

        # 環境から得た報酬を保存します。
        rewards.append(reward)

        # エピソード終了フラグを保存します。
        dones.append(done)

        # rollout時点の古い方策のlog probabilityを保存します。
        old_log_probs.append(log_prob.detach())

        # rollout時点のCriticの予測値を保存します。
        values.append(value.detach())

        if done:
            # 1エピソード終わっても続ける
            state, info = env.reset()
            state = torch.as_tensor(state,
                                    dtype=torch.float32,
                                    device=device)
        else:
            state = next_state

    # rollout終了後
    with torch.no_grad():
        _, next_value = model(state)
        next_value = next_value.squeeze(-1)

    states = torch.stack(states)
    actions = torch.stack(actions)
    values = torch.stack(values)
    old_log_probs = torch.stack(old_log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # 最後の時刻から逆向きに計算します。
    gae = 0.0
    for t in reversed(range(rollout_steps)):
        # 最後のstepでは rollout の外側で計算した next_value を使います。
        if t == rollout_steps - 1:
            next_v = next_value
        else:
            # それ以外では、次時刻の保存済みvalueを使います。
            next_v = values[t + 1]

        # TD errorを計算します。
        delta = rewards[t] + gamma * (1 - dones[t]) * next_v - values[t]

        # GAEを再帰的に後ろから計算します。
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        # 現在時刻のAdvantageを保存します。
        advantages.append(gae)

    # リスト全体を逆順にします。
    advantages = advantages[::-1]
    advantages = torch.stack(advantages)

    # Criticが学習する固定ターゲットを作ります。
    value_targets = advantages + values

    # Advantageを平均0・標準偏差1程度になるように標準化します。
    advantages = (advantages - advantages.mean()) / \
        (advantages.std() + 1e-8)
    '''
        ここまでrolloutによるデータ集め
    '''

    '''
        ここからミニバッチごとのPPO更新
    '''
    data_size = states.shape[0]
    for _ in range(ppo_epochs):

        # 0~2047をランダムに並べ替えた配列permutation
        indices = torch.randperm(data_size)

        # mini-batchの先頭位置を0からmini_batch_size刻みで動かします。
        for start in range(0, data_size, batch_size):
            # 現在のmini-batchの終了位置を求めます。
            end = start + batch_size

            # シャッフル済みのindexから、今回使うサンプル番号を取り出します。
            mb_indices = indices[start:end]

            '''
                ここからPPOの更新に必要なものをbufferから取り出す
            '''
            # 今回のmini-batchに対応する状態を取り出します。
            mb_states = states[mb_indices]

            # rollout時に実行した行動を取り出します。
            mb_actions = actions[mb_indices]

            # rollout時の古い方策のlog probabilityを取り出します。
            mb_old_log_probs = old_log_probs[mb_indices]

            # Actor更新用の標準化済みAdvantageを取り出します。
            mb_advantages = advantages[mb_indices]

            # Critic更新用の固定value targetを取り出します。
            mb_value_targets = value_targets[mb_indices]
            '''
                ここまで
            '''

            '''
                ここからLossの計算
            '''
            # 現在の方策π_newと現在のCriticの予測値を取得します。
            dist, new_values = model(mb_states)

            # Critic出力を (batch, 1) → (batch,) に変換します。
            new_values = new_values.squeeze(-1)

            # 保存してあるactionsを、現在の方策がどの程度出しやすいか計算します。
            new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

            # 保存してあるactionsが昔と比べてどのくらい出しやすくなったのかの比率
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # 通常のPolicy Gradient側の目的関数 r*A を計算します。
            surr1 = ratio * mb_advantages

            # ratioをクリップしたsurrogate objectiveを計算します。
            surr2 = torch.clamp(ratio,
                                1 - clip_eps,
                                1 + clip_eps
                                ) * mb_advantages

            # PPOのclipped objectiveを最小化用Lossへ変換します。
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = torch.nn.functional.mse_loss(
                new_values, mb_value_targets)

            # 各行動次元のEntropyを足し合わせ、
            # さらにmini-batch全体で平均を取ります。つまりスカラーになる
            entropy = dist.entropy().sum(dim=-1).mean()

            # Actor loss、Critic loss、Entropy bonusを全て足し算して
            # 最小化対象となる最終Lossを作ります。
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            '''
                ここまでLossの計算
            '''

            # 前回の更新で残っている勾配をゼロにします。
            optimizer.zero_grad()

            # PPOのLossから各パラメータの勾配を計算します。
            loss.backward()

            # 勾配ノルムが大きくなりすぎるのを防ぎます。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # 計算された勾配を使ってパラメータを更新します。
            optimizer.step()
    '''
        ここまでミニバッチごとのPPO更新
    '''
