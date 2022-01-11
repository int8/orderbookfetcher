from torch.utils.data import Dataset


class OrderBooksDataset(Dataset):

    def __init__(self, order_books_dir_path: str):
        self.order_books_dir_path = order_books_dir_path

    def __getitem__(self, index):
        pass


def __len__(self):
    return len(self.img_labels)
