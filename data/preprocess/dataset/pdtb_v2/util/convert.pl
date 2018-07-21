#!/usr/bin/perl

###############
#
# This Perl script converts the Penn Discourse Treebank (PDTB) corpus into a pipe-delimited format, with every
# relation represented on a single line, and every column fixed for a specific piece of data.  See the 
# accompanying COLUMN_DEFINITIONS file for column definitions.
#
# The script is meant to aid users who might not necessarily want to make use of the full capacity of the 
# Java-based APIs provided for the PDTB corpus (available at http://www.seas.upenn.edu/~pdtb).
# 
# The pipe-delimited output files may be imported into a spreadsheet file or users may choose to do further
# manipulation of the data using scripting programs.
#
# USAGE:
# 
# This is a Perl 5 script.
# It was written for a Unix/Linux environment, but users may try to use Perl for Win32 by changing the 
# $fileSeparator variable from a forward slash "/" to a backslah "\".  Note: This script has not been
# tested in a Windows environment.
#
# To run, go to the directory containing the script and type:
#
# ./convert.pl [pdtbRoot] [outputRoot]
# 
# where pdtbRoot corresponds to the root directory containing the pdtb annotation files (normally /data/pdtbRoot)
# and outputRoot is the root directory where the output files will be written into.
#
# outputRoot will mirror the structure of pdtbRoot, except that files will have a .pipe extension instead of the
# standard .pdtb extension.
#
# Once again, consult the accompanying COLUMN_DEFINITIONS file for column definitions.
###############


use strict;

our($pdtbRoot, $outputRoot, $fileSeparator);

$pdtbRoot = "";
$outputRoot = "";
$fileSeparator = "/";

if (@ARGV == 2) {
  &checkPdtbRoot($ARGV[0]);
  &checkOutputRoot($ARGV[1]);
} 
else {
  &printUsage();
}

for (my $i=0; $i<25; $i++) {
  for (my $j=0; $j<100; $j++) {
    my $secNo = appendZero($i);
    my $fileNo = appendZero($j);
    my $fileName = "wsj"."_".$secNo.$fileNo.".pdtb";
    my $fullPath = $pdtbRoot.$secNo.$fileSeparator.$fileName;

    my $writeSectionPath = $outputRoot.$secNo.$fileSeparator;
    unless (-d $writeSectionPath) {
      my $command = "mkdir $writeSectionPath";
      if (system($command)) {
	print "Failed to create directory $writeSectionPath\n";
	exit(0);
      }
    }

    my @tokens = ();
    &tokenize(\@tokens, $fullPath);

    my $count = @tokens;

    if ($count > 0) {
      my $writeFile = "wsj"."_".$secNo.$fileNo.".pipe";
      my $writeFilePath = $writeSectionPath.$writeFile;
      open WR, ">$writeFilePath";

      for (my $k=0; $k<$count; $k++) {
	my $tempRef = $tokens[$k];
	if ($$tempRef[0] eq "____Explicit____") {
	  my $output = &getExplicitString($tempRef, $secNo, $fileNo);
	  print WR $output;
	}
	elsif ($$tempRef[0] eq "____Implicit____") {
	  my $output = &getImplicitString($tempRef, $secNo, $fileNo);
	  print WR $output;
	}
	elsif ($$tempRef[0] eq "____AltLex____") {
	  my $output = &getAltLexString($tempRef, $secNo, $fileNo);
	  print WR $output;
	}
	elsif ($$tempRef[0] eq "____EntRel____") {
	  my $output = &getEntRelString($tempRef, $secNo, $fileNo);
	  print WR $output;
	}
	elsif ($$tempRef[0] eq "____NoRel____") {
	  my $output = &getNoRelString($tempRef, $secNo, $fileNo);
	  print WR $output;
	}
      }
      close(WR);
    }
  }
}



sub printUsage() {
  print "USAGE:  convert.pl [pdtbRoot] [outputRoot]\n";
  exit(0);
}

sub checkPdtbRoot {
  my ($r) = @_;

  if (-d $r) {
    $pdtbRoot = $r;
    $pdtbRoot =~ /.*(.)$/;
    if ($1 ne $fileSeparator) {
      $pdtbRoot .= $fileSeparator;
    }
    return 1;
  }
  else {
    print "Cannot find pdtbRoot directory\n";
    exit(0);
  }
}

sub checkOutputRoot {
  my ($r) = @_;

  if (-d $r) {
    $outputRoot = $r;
  }
  else {
    my $command = "mkdir $r";
    if (system($command)) {
      print "Failed to create outputRoot directory\n";
      exit(0);
    }
    else {
      $outputRoot = $r;
    }
  }

  $outputRoot =~ /.*(.)$/;
  if ($1 ne $fileSeparator) {
    $outputRoot .= $fileSeparator;
  }
  return 1;
}

sub appendZero {
  my ($a) = @_;
  if ($a < 10) {
    $a = "0".$a;
  }
  return $a;    
}

sub tokenize {
  my ($tokensRef, $path) = @_;
  my @current = ();

  open(FH, $path);
  my $indexTokens = 0;
  my $indexCurrent = 0;
  my $delimiter = 0;

  while (<FH>) {
    chomp($_);
    my $string = $_;

    if ($string eq "____Explicit____" || $string eq "____Implicit____" || $string eq "____AltLex____" ||
        $string eq "____EntRel____" || $string eq "____NoRel____") {
      $$tokensRef[$indexTokens][$indexCurrent] = $string;
      $indexCurrent++;
    }
    elsif ($string eq "________________________________________________________") {
      $delimiter++;
      if ($delimiter == 2) {
	$indexTokens++;
	$delimiter = 0;
	$indexCurrent = 0;
      }
    }
    else {
      $$tokensRef[$indexTokens][$indexCurrent] = $string;
      $indexCurrent++;
    }    
  } 
  close(FH);
}

sub getExplicitString {
  my ($tokenRef, $section, $file) = @_;

  my $offset = 0;
  $offset++;

  my $connSpanListGornRawText = &getSpanListGornRawText($tokenRef, \$offset);
  my $connFeatures = &getExplicitFeatures($tokenRef, \$offset);
  my $argsSups = &getArgsSups($tokenRef, \$offset);

  my $output = "Explicit|$section|$file|$connSpanListGornRawText|||$connFeatures|$argsSups\n";  
#  print "TEST: $output\n";
  return $output;
}

sub getImplicitString {
  my ($tokenRef, $section, $file) = @_;

  my $offset = 0;
  $offset++;

  my $connStringPosSentenceNo = &getStringPosSentenceNo($tokenRef, \$offset);
  my $connFeatures = &getImplicitFeatures($tokenRef, \$offset);
  my $argsSups = &getArgsSups($tokenRef, \$offset); 

  my $output = "Implicit|$section|$file||||$connStringPosSentenceNo|$connFeatures|$argsSups\n";  
#  print "TEST: $output\n";
  return $output;
}

sub getAltLexString {
  my ($tokenRef, $section, $file) = @_;

  my $offset = 0;
  $offset++;

  my $altLexSpanListGornRawText = getSpanListGornRawText($tokenRef, \$offset);
  my $altLexFeatures = getAltLexFeatures($tokenRef, \$offset);
  my $argsSups = &getArgsSups($tokenRef, \$offset);

  my $output = "AltLex|$section|$file|$altLexSpanListGornRawText|||$altLexFeatures|$argsSups\n";  
#  print "TEST: $output\n";
  return $output;
}

sub getEntRelString {
  my ($tokenRef, $section, $file) = @_;

  my $offset = 0;
  $offset++;

  my $stringPosSentenceNo = getStringPosSentenceNo($tokenRef, \$offset);
  my $args = &getArgs($tokenRef, \$offset);

  my $output = "EntRel|$section|$file||||$stringPosSentenceNo|||||||||||||||$args\n";  
#  print "TEST: $output\n";
  return $output;
}

sub getNoRelString {
  my ($tokenRef, $section, $file) = @_;

  my $offset = 0;
  $offset++;

  my $stringPosSentenceNo = getStringPosSentenceNo($tokenRef, \$offset);
  my $args = &getArgs($tokenRef, \$offset);

  my $output = "NoRel|$section|$file||||$stringPosSentenceNo|||||||||||||||$args\n";  
#  print "TEST: $output\n";
  return $output;
}

sub getArgsSups {
  my ($tokenRef, $offsetRef) = @_;

  my $sup1 = "||";
  if (&hasSup1($tokenRef, $offsetRef)) {
    $sup1 = &getSup($tokenRef, $offsetRef);
  }
  my $arg1 = &getArg($tokenRef, $offsetRef);
  my $arg2 = &getArg($tokenRef, $offsetRef);
  my $sup2 = "||";

  if (&hasSup2($tokenRef, $offsetRef)) {
    $sup2 = &getSup($tokenRef, $offsetRef);
  }

  my $output = "$arg1|$arg2|$sup1|$sup2";
  return $output;
}

sub getArgs {
  my ($tokenRef, $offsetRef) = @_;

  my $sup1 = "||";
  my $arg1 = &getArg($tokenRef, $offsetRef);
  my $arg2 = &getArg($tokenRef, $offsetRef);
  my $sup2 = "||";

  my $output = "$arg1|$arg2|$sup1|$sup2";
  return $output;
}

sub getArg{
  my ($tokenRef, $offsetRef) = @_;
  
  while ($$tokenRef[$$offsetRef] ne "____Arg1____" && $$tokenRef[$$offsetRef] ne "____Arg2____") {    
    $$offsetRef++;
  }
  $$offsetRef++;
  my $spanListGorn = getSpanListGorn($tokenRef, $offsetRef);
  my $rawText = getRawText($tokenRef, $offsetRef);
  my $attr = getAttribution($tokenRef, $offsetRef);

  my $output = "$spanListGorn|$rawText|$attr";
  return $output;
}

sub hasSup1 {
  my ($tokenRef, $offsetRef) = @_;

  my $regex = $$tokenRef[$$offsetRef];
  until ($regex eq "____Arg1____" || $regex eq "____Sup1____") {
    $$offsetRef++;
    $regex = $$tokenRef[$$offsetRef];
  }  

  if ($regex eq "____Arg1____") {
    return 0;
  }
  return 1;
}

sub hasSup2 {
  my ($tokenRef, $offsetRef) = @_;

  my $regex = $$tokenRef[$$offsetRef];
  until ($regex eq "" || $regex eq "____Sup2____") {
    $$offsetRef++;
    $regex = $$tokenRef[$$offsetRef];
  }  

  if ($regex eq "") {
    return 0;
  }
  return 1;
}

sub getSup {
  my ($tokenRef, $offsetRef) = @_;
  my $spanListGorn = "";
  my $rawText = "";

  if ($$tokenRef[$$offsetRef] eq "____Sup1____" || $$tokenRef[$$offsetRef] eq "____Sup2____") {
    $$offsetRef++;
    $spanListGorn = getSpanListGorn($tokenRef, $offsetRef);
    $rawText = getRawText($tokenRef, $offsetRef);
  }

  my $output = "$spanListGorn|$rawText";
  return $output;
}

sub getSpanListGornRawText {
  my ($tokenRef, $offsetRef) = @_;
  
  my $spanListGorn = getSpanListGorn($tokenRef, $offsetRef);
  my $rawText = getRawText($tokenRef, $offsetRef);
  my $string = "$spanListGorn|$rawText";
  return $string;
}

sub getSpanListGorn {
  my ($tokenRef, $offsetRef) = @_;

  my $spanList = $$tokenRef[$$offsetRef];
  $$offsetRef++;
  my $gorn = $$tokenRef[$$offsetRef];
  $$offsetRef++;
  my $string = "$spanList|$gorn";
  return $string;
}

sub getStringPosSentenceNo {
  my ($tokenRef, $offsetRef) = @_;

  my $stringPos = $$tokenRef[$$offsetRef];
  $$offsetRef++;
  my $sentenceNo = $$tokenRef[$$offsetRef];
  $$offsetRef++;
  my $string = "$stringPos|$sentenceNo";
  return $string;
}

sub getRawText {
  my ($tokenRef, $offsetRef) = @_;
  
  my $string = "";
  while(1) {
    if ($$tokenRef[$$offsetRef] eq "#### Text ####") {
      $$offsetRef++;
    }
    elsif ($$tokenRef[$$offsetRef] ne "##############") {      
      $string .= $$tokenRef[$$offsetRef];
      $$offsetRef++;
    }
    else {
      $string =~ s/\n//g;
      $$offsetRef++;
      return $string;
    }
  }
}


sub getExplicitFeatures {
  my ($tokenRef, $offsetRef) = @_;
  
  my $attribution = &getAttribution($tokenRef, $offsetRef);
  $$offsetRef++;
  
  my $headSenseString = $$tokenRef[$$offsetRef];
  my @headSenseArray = split(/,/, $headSenseString);
  my $connHead = &trim($headSenseArray[0]);
  my $sense1 = &trim($headSenseArray[1]);
  my $sense2 = "";
  if (@headSenseArray > 2) {
    $sense2 = &trim($headSenseArray[2]);
  }

  my $string = $connHead."|||".$sense1."|".$sense2."|||".$attribution;
  return $string;
}

sub getImplicitFeatures {
  my ($tokenRef, $offsetRef) = @_;
  
  my $attribution = &getAttribution($tokenRef, $offsetRef);
  $$offsetRef++;

  my $connSenseString = $$tokenRef[$$offsetRef];
  my @connSenseArray = split(/,/, $connSenseString);
  my $conn1 = &trim($connSenseArray[0]);
  my $sense1 = &trim($connSenseArray[1]);
  my $sense2 = "";
  if (@connSenseArray > 2) {
    $sense2 = &trim($connSenseArray[2]);
  }

  $$offsetRef++;
 
  my $regex = $$tokenRef[$$offsetRef];

  my $conn2 = "";
  my $sense3 = "";
  my $sense4 = "";

  if ($regex  =~ /.*,.*/) { 
    $connSenseString = $$tokenRef[$$offsetRef];
    @connSenseArray = split(/,/, $connSenseString);
    $conn2 = &trim($connSenseArray[0]);
    $sense3 = &trim($connSenseArray[1]);
    $sense4 = "";
    if (@connSenseArray > 2) {
      $sense4 = &trim($connSenseArray[2]);
    }    
  }

  my $string ="|".$conn1."|".$conn2."|".$sense1."|".$sense2."|".$sense3."|".$sense4."|".$attribution;
  return $string;
}

sub getAltLexFeatures {
  my ($tokenRef, $offsetRef) = @_;
  
  my $attribution = getAttribution($tokenRef, $offsetRef);
  $$offsetRef++;
  
  my $senseString = $$tokenRef[$$offsetRef];
  my @senseArray = split(/,/, $senseString);
  my $sense1 = &trim($senseArray[0]);
  my $sense2 = "";
  if (@senseArray > 2) {
    $sense2 = &trim($senseArray[1]);
  }

  my $string = "|||".$sense1."|".$sense2."|||".$attribution;
  return $string;
}



sub getAttribution {
  my ($tokenRef, $offsetRef) = @_;

  if ($$tokenRef[$$offsetRef] eq "#### Features ####") {
    $$offsetRef++;
    
    my $attrString = $$tokenRef[$$offsetRef];
    my @attrArray = split(/,/, $attrString);
    my $source = &trim($attrArray[0]);
    my $type = &trim($attrArray[1]);
    my $pol = &trim($attrArray[2]);
    my $det = &trim($attrArray[3]);
    my $temp1 = $source."|".$type."|".$pol."|".$det;
    $$offsetRef++;
  
    my $regex = $$tokenRef[$$offsetRef];
    my $temp2 = "||";
    if ($regex  =~ /^\d+\.\.\d+/) {
      $temp2 = getSpanListGornRawText($tokenRef, $offsetRef);
      my $string = $temp1."|".$temp2;
      --$$offsetRef;
      return $string; 
    } 
    else {
      my $string = $temp1."|".$temp2;
      --$$offsetRef;
      return $string;
    }
  }
  return "||||||";
}

sub trim {
  my ($string) = @_;

  for ($string) {
    s/^\s+//;
    s/\s+$//;
  }
  return $string;
}
