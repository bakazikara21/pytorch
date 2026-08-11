# ファイルやディレクトリのパスを安全に扱うためのPathクラスを読み込みます。
from pathlib import Path

# ファイルを別のディレクトリへ移動するための標準ライブラリを読み込みます。
import shutil

# フォルダの選択ダイアログ
# pip install pysimplegui
import PySimpleGUI as sg


def get_target_directory() -> Path | None:
    # 対象のディレクトリを選ばせるstring型
    target_directory = sg.popup_get_folder(
        message="ファイルを整理したい対象のディレクトリを選んでください。",
        title="Select a directory",
        no_window=True,
        # size=(800, 600),
    )

    # キャンセルされた場合はNoneを返します。
    if target_directory is None:
        return None

    # 文字列のパスをPathオブジェクトへ変換して返します。
    return Path(target_directory)


def get_extension_folder_name(file_path: Path) -> str:
    """
    ファイルの拡張子から、振り分け先のフォルダ名を決定します。

    拡張子が存在しないファイルは、no_extensionフォルダへ分類します。
    """

    # ファイルの拡張子を取得し、小文字へ統一します。e.g. .pdf, .txt
    extension = file_path.suffix.lower()

    # 拡張子が存在しない場合のフォルダ名を返します。
    if not extension:
        return "no_extension"

    # 拡張子の先頭に付いているドットを削除して返します。
    return extension.lstrip(".")


def get_unique_destination(destination: Path) -> Path:
    """
    移動先に同名ファイルが存在する場合、
    上書きを防ぐために連番付きのファイル名を生成します。

    例:
        report.pdf
        report_1.pdf
        report_2.pdf
    """

    # 移動先に同名ファイルが存在しない場合は、そのパスをそのまま返します。
    if not destination.exists():
        return destination

    # ファイル名から拡張子を除いた部分を取得します。
    file_stem = destination.stem

    # ファイル名の拡張子を取得します。
    file_suffix = destination.suffix

    # 同名ファイルを区別するための連番を1から開始します。
    counter = 1

    # 使用可能なファイル名が見つかるまで繰り返します。
    while True:
        # 「元の名前_連番.拡張子」という新しいファイル名を作成します。
        new_file_name = f"{file_stem}_{counter}{file_suffix}"

        # 元の移動先ディレクトリに、新しいファイル名を結合します。
        new_destination = destination.parent / new_file_name

        # 新しいファイル名がまだ使用されていなければ、そのパスを返します。
        if not new_destination.exists():
            return new_destination

        counter += 1


def organize_files(target_directory: Path, dry_run: bool = False) -> None:
    """
    指定されたディレクトリ直下のファイルを、拡張子ごとのフォルダへ移動します。

    Parameters
    ----------
    target_directory:
        整理対象となるディレクトリです。

    dry_run:
        Trueの場合は、実際にはファイルを移動せず、
        実行予定の操作だけを表示します。
    """

    # ユーザーが指定したパスを絶対パスへ変換します。
    target_directory = target_directory.expanduser().resolve()

    # 指定されたパスが存在するか確認します。
    if not target_directory.exists():
        # 存在しない場合は、原因が分かるように例外を発生させます。
        raise FileNotFoundError(
            f"指定されたディレクトリが存在しません: {target_directory}"
        )

    if not target_directory.is_dir():
        # ファイルなどを指定してしまった場合、
        raise NotADirectoryError(
            f"指定されたパスはディレクトリではありません: {target_directory}"
        )

    # 対象ディレクトリ直下にある要素を1つずつ確認します。
    for item in target_directory.iterdir():
        # ディレクトリは整理対象にせず、次の要素へ進みます。
        if item.is_dir():
            continue

        # シンボリックリンクは意図しない移動を防ぐため対象外にします。
        if item.is_symlink():
            print(f"[スキップ] シンボリックリンク: {item.name}")
            continue

        # 通常のファイルでない要素は対象外にします。
        if not item.is_file():
            continue

        # ファイルの拡張子に応じたフォルダ名を取得します。
        folder_name = get_extension_folder_name(item)

        # 対象ディレクトリとフォルダ名を結合し、振り分け先を決定します。
        destination_directory = target_directory / folder_name

        # 元のファイル名を維持した移動先パスを作成します。
        destination = destination_directory / item.name

        # 同名ファイルが存在する場合は、重複しない名前へ変更します。
        destination = get_unique_destination(destination)

        # 振り分け先のディレクトリが存在しない場合は作成します。
        # 絶対パスの親ディレクトリも存在しなければ作成
        # 元々ディレクトリが存在していてもエラーにしない
        destination_directory.mkdir(parents=True, exist_ok=True)

        # ファイルを決定した移動先へ移動します。
        shutil.move(str(item), str(destination))

        # 実行した処理をユーザーが確認できるように表示します。
        print(f"[移動] {item.name} -> {destination}")


def main() -> None:
    """
    プログラム全体の処理を開始するメイン関数です。
    """
    target_directory: Path = get_target_directory()

    organize_files(target_directory)


# このファイルが直接実行された場合だけ、main関数を呼び出します。
if __name__ == "__main__":
    # プログラムの処理を開始します。
    main()
