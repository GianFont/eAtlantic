import numpy as np
import os

#check physical distance is respected

class Env:

    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.network = np.full((self.width, self.height), 0)

    def step(self, col_idx):
    ### function to move from one state to the next
    # state is the current state s_0 of the network - the positions of the colored discs in the suspended grid

        # invalid column index is provided
        if col_idx >= self.width:
            state = self.network.copy()
            # reward = -100
            reward = -1
            result = -1
            return state, reward, result

        # the provided column is full
        row_idx = np.argmin(self.network, 1)[col_idx]
        if self.network[col_idx, row_idx] != 0:
            state = self.network.copy()
            # reward = -100
            reward = -1
            result = -1
            return state, reward, result

        # the provided column is valid
        # drop the chip to this column
        self.network[col_idx, row_idx] = step_player

        state = self.network.copy()
        result = self._judge(col_idx, row_idx)
        if result == step_player:
            reward = 1
        else:
            reward = 0

        # print(reward)

        return state, reward, result

    def simulate(self, test_state, col_idx):
        # backup the current state and current player
        snapshot = self.network.copy()
        # play on the given state
        self.network = test_state.copy()
        state, reward, result = self.step(col_idx)
        # restore the state and current player
        self.network = snapshot

        return state, reward, result

    def _judge(self, col_idx, row_idx):
        result = 0
        player = self.network[col_idx, row_idx]

        # vertical
        if row_idx >= 3:
            check_range = self.network[col_idx, np.arange(row_idx - 3, row_idx + 1)]
            if len(check_range[check_range != player]) == 0:
                result = player
                # print('Player', player, 'won vertically!')

        # horizontal
        for idx in range(col_idx - 3, col_idx + 4):
            if 0 <= idx < self.width:
                new_range = np.arange(max(idx, idx - 3), min(self.width, idx + 4))
                if len(new_range) >= 4:
                    check_range = self.network[new_range, row_idx]
                    if len(check_range[check_range != player]) == 0:
                        result = player
                        # print('Player', player, 'won horizontally!')
                        break
        # diagonal
        diag_bwd = np.diagonal(np.rot90(self.network), offset=((row_idx + 1) - (self.width - (col_idx + 1))))
        diag_fwd = np.diagonal(np.flipud(np.rot90(self.network)), offset=col_idx - row_idx)
        # print(diag_bwd)
        # print(diag_fwd)
        if len(diag_bwd) >= 4:
            for idx in range(0, len(diag_bwd) - 3):
                check_range = diag_bwd[idx:idx + 4]
                # print(idx, idx + 4)
                if len(check_range[check_range != player]) == 0:
                    result = player
                    # print('Player', player, 'won backward diagonally!')
                    break
        if len(diag_fwd) >=4:
            for idx in range(0, len(diag_fwd) - 3):
                check_range = diag_fwd[idx:idx + 4]
                # print(idx, idx + 4)
                if len(check_range[check_range != player]) == 0:
                    result = player
                    # print('Player', player, 'won forward diagonally!')
                    break
        # draw
        if len(self.network.reshape(self.width * self.height).nonzero()[0]) == self.width * self.height:
            # print('Draw game!')
            result = 3

        return result

    def get_all_next_actions(self):
        return [action for action in range(self.width)]

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.network.copy()
        actions = []
        for col_idx in range(self.width):
            if np.min(state, 1)[col_idx] > 0:
                actions.append(0)
            else:
                actions.append(1)
        return actions

    def reset(self):
        self.network = np.full((self.width, self.height), 0)

    def to_str(self, network=None):
        string = os.linesep
        if network is None:
            network = self.network
        b = np.rot90(network).reshape(self.width * self.height)
        for idx, c in enumerate(b):
            c = int(c)
            if (idx + 1) % self.width > 0:
                string = '{}{} '.format(string, c)
            else:
                string = '{}{}{}'.format(string, c, os.linesep)
        return string

    def print(self, network=None):
        print(self.to_str(network))

    def get_state(self):
        return self.network.copy().astype(dtype=np.float32)

    def get_mirror_state(self, network=None):
        if network is None:
            network = self.network
        mirror = np.array(network)
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                mirror[self.width - col_idx - 1, row_idx] = network[col_idx, row_idx]
        # return mirror.astype(dtype=np.float32)
        return mirror

    def get_inv_state(self, network=None):
        if network is None:
            network = self.network
        # inv = network.copy().astype(dtype=np.float32)
        inv = network.copy()
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                if inv[col_idx, row_idx] == 1:
                    inv[col_idx, row_idx] = 2
                elif inv[col_idx, row_idx] == 2:
                    inv[col_idx, row_idx] = 1
        return inv



def main():
    b = Connect4Env()

    while True:
        b.print()
        col_idx = int(input('Player {}\'s turn. Please input the col number (1 to {}) you want to place your chip:'.format(b.get_current_player(), b.width)))
        state, reward, result = b.step(col_idx - 1)
        if result < 0:
            print('Your input is invalid.')
        elif result == 0:
            pass
        elif result == 3:
            print('Draw game!!!!!')
        else:
            print('Player', b.get_current_player(), 'won!!!!!')
            b.print()
            break


if __name__ == '__main__':
    main()
