import csv


class CSVLogger:
    def __init__(self, filename: str, fields: list[str]) -> None:
        self.filename = filename
        self.fields = fields

        self._initalize_header()

    def _initalize_header(self) -> None:
        try:
            with open(self.filename, mode="x") as file_handle:
                writer = csv.writer(file_handle)
                writer.writerow(self.fields)
        except FileExistsError:
            pass

    def log(self, **kwargs) -> None:
        with open(self.filename, mode="a") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=self.fields)
            writer.writerow(kwargs)
