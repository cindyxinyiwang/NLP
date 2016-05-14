for file in *.out
do
   #echo "file: $file"
   #echo -n "first line: "
   #grep -v '^\s*$' "$file" | head -n1
   #echo -n "last line: "
   grep -v '^\s*$' "$file" | tail -2
done
