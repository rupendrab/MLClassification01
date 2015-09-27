my_download <- function(url, filename) {
    if (! file.exists(filename)) {
        download.file(url=url, destfile=filename, method="curl")
    }
}

my_download("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "data/pml-training.csv")
my_download("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "data/pml-testing.csv")
