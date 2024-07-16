import sys, shutil, math

def clearKLines(k):
    for _ in range(k):
        sys.stdout.write('\033[F')
        sys.stdout.write('\033[K')


class LoadingBar():
    def __init__(self, steps: int, title: str = "Running process...", extraInfos: list = []):
        self.steps = steps
        self.title = title
        self.extraInfos = extraInfos
        self.prevWidth = -1

    def start(self):
        width = shutil.get_terminal_size().columns
        titleMsg = self.title + " " * (int(width * .85) - len(self.title) - 2)

        print('\n', titleMsg, '\n' * (len(self.extraInfos) + 1))
        self.prevWidth = width
        self.updateBar(0)

    def updateBar(self, crrStep: int):
        width, _ = shutil.get_terminal_size()
        clearKLines(math.ceil((self.prevWidth * .85) / width) + len(self.extraInfos))
        self.prevWidth = width

        crrPerc = crrStep / self.steps

        leftSideBar = "Loading: |"
        rightSideBar = f"| {(crrPerc * 100):.2f}% ({crrStep}/{self.steps})"

        emptySpace = width * .85 - len(leftSideBar) - len(rightSideBar)
        barPorgress = math.ceil(emptySpace * crrPerc) * "#"
        emptySection = math.floor((emptySpace * (1 - crrPerc))) * "-"

        print(leftSideBar + barPorgress + emptySection + rightSideBar)

        for line in self.extraInfos:
            leftSpace = int((width * .85) - len(line)) * " "
            print(leftSpace + line)
