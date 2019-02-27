cd data
for f in *bz2;
do
    bunzip2 $f;
done
